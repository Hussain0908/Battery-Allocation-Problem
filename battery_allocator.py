"""
Battery Allocation Optimization for Electric Buses using QAOA
==============================================================
This script demonstrates converting a QUBO matrix to an Ising Hamiltonian
and solving it using QAOA.

Problem: Multiple electric buses that distribute energy. We want to install
a total of K batteries distributed among the buses. Each bus can have
0, 1, 2, ..., up to max_batteries batteries. The energy saving at each
bus depends on the number of batteries installed there.

QUBO Formulation (based on Multi-Knapsack Problem approach):
- Binary variables: x_{i,j} = 1 if bus i gets exactly j batteries
- Objective: Maximize total energy savings (or minimize negative savings)
- Constraints:
  1. Each bus gets exactly one choice: sum_j x_{i,j} = 1 for all i
  2. Total batteries = K: sum_i sum_j (j * x_{i,j}) = K

Reference: QUBO Formulations for Multi-Knapsack Problem
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10924197
"""

import time
from itertools import product

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# =============================================================================
# Step 1: Define the problem parameters
# =============================================================================

# Energy savings matrix: savings[i][j] = savings at bus i with j batteries
# Row i = bus i, Column j = number of batteries (0 to max_batteries)
# Note: savings[i][0] = 0 (no batteries = no savings)
#
# IBM Fez has 156 qubits. With one-hot encoding: n_buses × (max_batteries + 1) ≤ 156
# For 51 buses with up to 2 batteries: 51 × 3 = 153 qubits
#
# Savings follow diminishing returns: savings = base * (1 - exp(-rate * j))
def generate_savings(base, rate, max_batt):
    """Generate diminishing returns savings curve."""
    return [0.0] + [round(base * (1 - np.exp(-rate * j)), 2) for j in range(1, max_batt + 1)]

# Generate savings for 51 buses with up to 2 batteries each
n_nodes = 3
max_batt_per_node = 4

# Create varied savings profiles for each bus
# Base savings range from 1000 to 2000, rate from 0.3 to 0.8
np.random.seed(123)  # For reproducibility
savings = [
    generate_savings(
        base=1000 + (i * 20),  # Base savings: 1000-2000 range
        rate=0.3 + (i % 10) * 0.05,  # Rate: 0.3-0.75 range
        max_batt=max_batt_per_node
    )
    for i in range(n_nodes)
]

# Infere values from the savings matrix to support custom ones
n_buses = len(savings)  # Number of electric buses
max_batteries = len(savings[0]) - 1  # Maximum batteries per bus (0, 1, or 2)
K = 5  # Total number of batteries to distribute (average 1 per bus)

# Penalty strengths for constraint violations
# According to Theorem 2 (MUKP): λ₁ ≥ R* and λ₂ ≥ R*
# where R* = max{savings[i][j]} = 991.97965906
# Paper reference: "QUBO Formulations and Characterization of Penalty Parameters
# for the Multi-Knapsack Problem" (Güney et al., IEEE Access 2025)
penalty = max(max(row) for row in savings)  # = 991.97965906
penalty_multiplier = 1  # Very conservative multiplier for 15-variable MUKP
lambda_one_choice = penalty_multiplier * penalty  # λ₂ (one-hot constraint per bus)
lambda_total = penalty_multiplier * penalty  # λ₁ (total batteries = K)

print("\n[Penalty Parameters]")
print(f"R* (max savings) = {penalty:.2f}")
print(f"Penalty multiplier = {penalty_multiplier}x")
print(f"λ₁ (total constraint) = {lambda_total:.2f}")
print(f"λ₂ (one-hot constraint) = {lambda_one_choice:.2f}")

# Installation costs per battery (can be bus-dependent)
c_install = 50  # Cost per battery installed

# =============================================================================
# Step 2: Build the QUBO matrix for Multi-Bus Battery Allocation
# =============================================================================


def build_battery_allocation_qubo(n_buses, max_batt, k_total, savings_matrix,
                                   c_install, lambda_one, lambda_k):
    """
    Construct the QUBO matrix for battery allocation across multiple buses.

    Variables: x_{i,j} where i = bus index, j = number of batteries (0 to max_batt)
    Total variables: n_buses * (max_batt + 1)

    Variable ordering: x_{0,0}, x_{0,1}, ..., x_{0,max}, x_{1,0}, x_{1,1}, ...

    Objective (to minimize):
        -sum_i sum_j savings[i][j] * x_{i,j} + c_install * sum_i sum_j j * x_{i,j}

    Constraints (as penalties):
        1. sum_j x_{i,j} = 1 for each bus i  =>  lambda_one * (sum_j x_{i,j} - 1)^2
        2. sum_i sum_j j * x_{i,j} = K       =>  lambda_k * (sum_i sum_j j*x_{i,j} - K)^2
    """
    n_choices = max_batt + 1  # 0, 1, 2, ..., max_batt
    n_vars = n_buses * n_choices

    # Validate savings matrix dimensions
    if len(savings_matrix) != n_buses:
        raise ValueError(
            f"savings_matrix must have {n_buses} rows (one per bus), "
            f"got {len(savings_matrix)}"
        )
    for i, row in enumerate(savings_matrix):
        if len(row) != n_choices:
            raise ValueError(
                f"savings_matrix[{i}] must have {n_choices} columns "
                f"(0 to {max_batt} batteries), got {len(row)}"
            )

    Q = np.zeros((n_vars, n_vars))

    def var_index(bus, num_batt):
        """Get variable index for x_{bus, num_batt}."""
        return bus * n_choices + num_batt

    # -------------------------------------------------------------------------
    # Part 1: Objective function (linear terms)
    # Minimize: -savings + installation_cost = sum_i sum_j (-s_{ij} + c*j) * x_{ij}
    # -------------------------------------------------------------------------
    for i in range(n_buses):
        for j in range(n_choices):
            idx = var_index(i, j)
            # Negative savings (we minimize, so negative = good)
            # Plus installation cost (j batteries cost c_install * j)
            Q[idx, idx] += -savings_matrix[i][j] + c_install * j

    # -------------------------------------------------------------------------
    # Part 2: Constraint 1 - Each bus gets exactly one choice
    # For each bus i: (sum_j x_{i,j} - 1)^2 = sum_j x_{i,j}^2 + 2*sum_{j<k} x_{i,j}*x_{i,k} - 2*sum_j x_{i,j} + 1
    # Since x_{i,j} is binary: x^2 = x, so:
    # = sum_j x_{i,j} + 2*sum_{j<k} x_{i,j}*x_{i,k} - 2*sum_j x_{i,j} + 1
    # = -sum_j x_{i,j} + 2*sum_{j<k} x_{i,j}*x_{i,k} + 1
    #
    # Coefficients:
    #   - Diagonal (linear): -1 for each x_{i,j}
    #   - Off-diagonal: +2 for each pair (x_{i,j}, x_{i,k}) with j < k
    # -------------------------------------------------------------------------
    for i in range(n_buses):
        # Linear terms
        for j in range(n_choices):
            idx = var_index(i, j)
            Q[idx, idx] += lambda_one * (-1)  # from -2*x + x^2 = -x for binary

        # Quadratic terms (pairs within same bus)
        for j in range(n_choices):
            for jj in range(j + 1, n_choices):
                idx_j = var_index(i, j)
                idx_jj = var_index(i, jj)
                Q[idx_j, idx_jj] += lambda_one * 2

    # -------------------------------------------------------------------------
    # Part 3: Constraint 2 - Total batteries = K
    # (sum_i sum_j j * x_{i,j} - K)^2
    #
    # Expand: sum_i sum_j j^2 * x_{i,j}^2
    #         + 2 * sum_{(i,j)<(i',j')} j*j' * x_{i,j} * x_{i',j'}
    #         - 2K * sum_i sum_j j * x_{i,j} + K^2
    #
    # Since x^2 = x for binary:
    #   - Diagonal: j^2 - 2Kj = j(j - 2K) for x_{i,j}
    #   - Off-diagonal: 2*j*j' for pairs x_{i,j}, x_{i',j'}
    # -------------------------------------------------------------------------
    for i in range(n_buses):
        for j in range(n_choices):
            idx = var_index(i, j)
            # Diagonal contribution from constraint 2
            Q[idx, idx] += lambda_k * j * (j - 2 * k_total)

    # Off-diagonal terms for constraint 2 (all pairs of variables)
    for i1 in range(n_buses):
        for j1 in range(n_choices):
            idx1 = var_index(i1, j1)
            for i2 in range(n_buses):
                for j2 in range(n_choices):
                    idx2 = var_index(i2, j2)
                    if idx1 < idx2:  # Only upper triangular
                        Q[idx1, idx2] += lambda_k * 2 * j1 * j2

    return Q, var_index


Q, var_index = build_battery_allocation_qubo(
    n_buses, max_batteries, K, savings, c_install, lambda_one_choice, lambda_total
)

n_choices = max_batteries + 1  # Number of choices per bus (0, 1, ..., max_batteries)
n_vars = n_buses * n_choices  # Total number of binary variables

print("=" * 70)
print("BATTERY ALLOCATION PROBLEM FOR ELECTRIC BUSES")
print("=" * 70)
print(f"Number of buses: {n_buses}")
print(f"Max batteries per bus: {max_batteries}")
print(f"Total batteries to distribute: {K}")
print(f"Total binary variables: {n_vars}")
print("\nSavings matrix (rows=buses, cols=num_batteries):")
for i, row in enumerate(savings):
    print(f"  Bus {i}: {row}")
print()
print("QUBO Matrix Q (shape: {0}x{0}):".format(n_vars))
print(Q)
print()

# =============================================================================
# Step 3: Create QuadraticProgram from QUBO matrix
# =============================================================================


def qubo_matrix_to_quadratic_program(Q, n_buses, max_batt):
    """
    Convert a QUBO matrix to a Qiskit QuadraticProgram.

    Variable naming: x_{bus}_{num_batteries}
    """
    n_choices = max_batt + 1

    # Create meaningful variable names
    var_names = []
    for i in range(n_buses):
        for j in range(n_choices):
            var_names.append(f"x_{i}_{j}")

    qp = QuadraticProgram(name="Battery Allocation")

    # Add binary variables with custom names
    for name in var_names:
        qp.binary_var(name=name)

    # Extract linear (diagonal) and quadratic (off-diagonal) terms from Q matrix
    # The minimize() method accepts numpy arrays directly
    linear = np.diag(Q)
    quadratic = Q - np.diag(np.diag(Q))  # Zero out diagonal, keep off-diagonal

    # Set objective (minimize) - pass numpy arrays directly
    qp.minimize(linear=linear, quadratic=quadratic)

    return qp, var_names


qp, var_names = qubo_matrix_to_quadratic_program(Q, n_buses, max_batteries)

print("Quadratic Program:")
print(qp.prettyprint())
print()

# =============================================================================
# Step 4: Convert to Ising Hamiltonian
# =============================================================================

# The QuadraticProgramToQubo converter ensures the problem is in QUBO form
# (ours already is, but this is the standard pipeline)
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

# Get the Ising Hamiltonian (operator) and offset
operator, offset = qubo.to_ising()

print("Ising Hamiltonian:")
print(operator)
print(f"\nOffset (constant term): {offset}")
print()

# =============================================================================
# Step 5: Run QAOA with Ising Hamiltonian
# =============================================================================

# Connect to IBM Quantum Runtime Service
# Make sure you have saved your credentials with:
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_API_TOKEN")
service = QiskitRuntimeService()

# Select a backend with enough qubits (need at least 153 for this problem)
# ibm_fez has 156 qubits, use it explicitly
backend = service.backend("ibm_fez")
print(f"Using backend: {backend.name} ({backend.num_qubits} qubits)")

# Set up QAOA with IBM Quantum sampler
# Generate a pass manager to transpile circuits to the backend's ISA
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

sampler = SamplerV2(backend)

# For IBM Runtime, we need to handle transpilation properly
# Use StatevectorSampler for local simulation (faster for development)
# Set USE_REAL_HARDWARE = True when ready to run on IBM Quantum hardware
# Note: Running on real hardware with the open plan has limitations:
# - Sessions are not supported (only single jobs or batch mode)
# - QAOA with many iterations will queue for a long time
# - Circuit transpilation + operator mapping is complex
# For development/testing, use the local simulator (USE_REAL_HARDWARE = False)
USE_REAL_HARDWARE = False

if USE_REAL_HARDWARE:
    # Running QAOA on real hardware requires a custom implementation that:
    # 1. Pre-transpiles the QAOA ansatz to ISA
    # 2. Remaps the operator to match the physical qubit layout
    # 3. Runs the optimization loop with properly transpiled circuits
    print("Using IBM Quantum hardware (this may take a while)...")
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
    from qiskit.circuit.library import QAOAAnsatz
    from scipy.optimize import minimize as scipy_minimize

    # Generate pass manager for transpilation
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)

    # Build the QAOA ansatz and add measurements
    reps = 2
    ansatz = QAOAAnsatz(operator, reps=reps)
    ansatz.measure_all()  # Add measurements to all qubits

    # Transpile the circuit
    transpiled_ansatz = pass_manager.run(ansatz)
    print(f"Transpiled ansatz: {transpiled_ansatz.num_qubits} qubits, depth {transpiled_ansatz.depth()}")

    # Get the layout mapping: virtual qubit -> physical qubit
    layout = transpiled_ansatz.layout
    virtual_bits = list(layout.initial_layout.get_virtual_bits().keys())
    physical_bits = [layout.initial_layout.get_virtual_bits()[q] for q in virtual_bits]
    print(f"Qubit mapping (virtual -> physical): {dict(enumerate(physical_bits[:15]))}")

    # Note: We don't need to remap the operator because:
    # - The circuit is transpiled to physical qubits
    # - But measure_all() measurements are returned in LOGICAL qubit order
    # - So we use the original operator for energy calculation

    # Create the sampler
    sampler = RuntimeSamplerV2(backend)

    # Custom QAOA optimization loop
    num_params = transpiled_ansatz.num_parameters
    print(f"Number of variational parameters: {num_params}")

    iteration_count = [0]

    def evaluate_energy(params):
        """Evaluate the expectation value of the Hamiltonian."""
        # Bind parameters to the transpiled ansatz
        bound_circuit = transpiled_ansatz.assign_parameters(params)

        # Run the circuit
        job = sampler.run([bound_circuit], shots=1024)
        result = job.result()

        # Get counts from the result - SamplerV2 returns BitArray
        # The classical register is named 'meas' from measure_all()
        pub_result = result[0]
        bit_array = pub_result.data.meas
        counts = bit_array.get_counts()

        # Calculate expectation value from measurement results
        total_shots = sum(counts.values())
        expectation = 0.0

        for bitstring, count in counts.items():
            # Convert bitstring to spin values (+1, -1)
            # Note: bitstring is in little-endian order from Qiskit
            # The measurement results are in logical qubit order (0-14), not physical
            spins = np.array([1 - 2 * int(b) for b in bitstring])

            # Calculate energy for this state using the ORIGINAL operator (logical qubits)
            # The measurements are returned in logical qubit order
            energy = 0.0
            for pauli, coeff in zip(operator.paulis, operator.coeffs):
                # Calculate <s|P|s> for diagonal Pauli P (only Z terms)
                pauli_val = 1.0
                for i, z_bit in enumerate(pauli.z):
                    if z_bit:
                        pauli_val *= spins[i]
                energy += coeff.real * pauli_val

            expectation += (count / total_shots) * energy

        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}: energy = {expectation:.4f}")

        return expectation

    # Run optimization
    print("Running QAOA optimization...")
    qaoa_start_time = time.time()

    # Initial parameters (random)
    np.random.seed(42)
    initial_params = np.random.uniform(-np.pi, np.pi, num_params)

    # Use COBYLA optimizer
    opt_result = scipy_minimize(
        evaluate_energy,
        initial_params,
        method='COBYLA',
        options={'maxiter': 6, 'rhobeg': 0.5}  # Reduced iterations for hardware
    )

    qaoa_elapsed_time = time.time() - qaoa_start_time

    # Get the final result with optimal parameters
    optimal_params = opt_result.x
    bound_circuit = transpiled_ansatz.assign_parameters(optimal_params)
    final_job = sampler.run([bound_circuit], shots=4096)
    final_result = final_job.result()
    final_bit_array = final_result[0].data.meas
    final_counts = final_bit_array.get_counts()

    # Find the most frequent bitstring
    # The bitstring is already in logical qubit order (not physical)
    best_bitstring = max(final_counts, key=final_counts.get)

    # Reverse for little-endian to match our convention
    best_bitstring = best_bitstring[::-1]
    result_x = np.array([int(b) for b in best_bitstring])

    # Create a mock result object for compatibility
    class MockQAOAResult:
        def __init__(self, eigenvalue, bitstring):
            self.eigenvalue = eigenvalue
            self.best_measurement = {"bitstring": bitstring}

    qaoa_result = MockQAOAResult(opt_result.fun, best_bitstring[::-1])

else:
    print("Using local StatevectorSampler for development...")
    sampler_to_use = StatevectorSampler()
    optimizer = COBYLA(maxiter=200)

    qaoa = QAOA(
        sampler=sampler_to_use,
        optimizer=optimizer,
        reps=2,  # Number of QAOA layers (p)
    )

    print("Running QAOA with Ising Hamiltonian...")
    qaoa_start_time = time.time()
    qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
    qaoa_elapsed_time = time.time() - qaoa_start_time

    # Extract the best solution from QAOA result
    best_bitstring = qaoa_result.best_measurement["bitstring"]
    # Reverse the bitstring (Qiskit uses little-endian ordering)
    best_bitstring = best_bitstring[::-1]
    # Convert bitstring to binary array
    result_x = np.array([int(b) for b in best_bitstring])
    # Convert bitstring to binary array
    result_x = np.array([int(b) for b in best_bitstring])

# Calculate the actual QUBO objective value
qubo_objective = float(result_x @ Q @ result_x)

print(f"\nQAOA eigenvalue (Ising): {qaoa_result.eigenvalue.real}")
print(f"Offset: {offset}")
print(f"QUBO objective (eigenvalue + offset): {qaoa_result.eigenvalue.real + offset}")

print("\n" + "=" * 70)
print("QAOA RESULTS")
print("=" * 70)
print(f"\nOptimal solution vector: {result_x}")
print(f"Objective value (QUBO): {qubo_objective}")

# =============================================================================
# Step 6: Interpret the solution
# =============================================================================


def interpret_solution(x, n_buses, max_batt, savings_matrix, c_install):
    """
    Interpret the binary solution vector.

    Returns allocation dict: {bus_id: num_batteries}
    """
    n_choices = max_batt + 1
    allocation = {}

    for i in range(n_buses):
        for j in range(n_choices):
            idx = i * n_choices + j
            if x[idx] == 1:
                allocation[i] = j
                break

    return allocation


allocation = interpret_solution(result_x, n_buses, max_batteries, savings, c_install)

print("\n--- Battery Allocation ---")
total_batteries = 0
total_savings = 0
total_cost = 0

for bus_id, num_batt in sorted(allocation.items()):
    sav = savings[bus_id][num_batt]
    cost = c_install * num_batt
    total_batteries += num_batt
    total_savings += sav
    total_cost += cost
    print(f"  Bus {bus_id}: {num_batt} batteries → Savings: ${sav}, Cost: ${cost}")

print(f"\nTotal batteries installed: {total_batteries}")
print(f"Total energy savings: ${total_savings}")
print(f"Total installation cost: ${total_cost}")
print(f"Net benefit (savings - cost): ${total_savings - total_cost}")

# Verify constraints
constraint1_ok = len(allocation) == n_buses  # Each bus has exactly one choice
constraint2_ok = total_batteries == K  # Total batteries = K

print("\n--- Constraint Verification ---")
if constraint1_ok:
    print("✓ Constraint 1: Each bus has exactly one allocation")
else:
    print("✗ Constraint 1 violated: Some buses have multiple/no allocations")

if constraint2_ok:
    print(f"✓ Constraint 2: Total batteries = {K}")
else:
    print(f"✗ Constraint 2 violated: {total_batteries} batteries (expected {K})")

# =============================================================================
# Step 7: Compare with brute force (for verification)
# =============================================================================

print("\n" + "=" * 70)
print("VERIFICATION: Brute Force Search")
print("=" * 70)


def evaluate_solution(z, Q):
    """Evaluate QUBO objective for a binary solution."""
    z = np.array(z)
    return z @ Q @ z


def is_valid_solution(z, n_buses, max_batt, k_total):
    """
    Check if solution satisfies both constraints:
    1. Each bus has exactly one choice (one-hot encoding per bus)
    2. Total batteries = K
    """
    n_choices = max_batt + 1

    # Check constraint 1: exactly one choice per bus
    for i in range(n_buses):
        bus_choices = z[i * n_choices : (i + 1) * n_choices]
        if sum(bus_choices) != 1:
            return False

    # Check constraint 2: total batteries = K
    total = 0
    for i in range(n_buses):
        for j in range(n_choices):
            idx = i * n_choices + j
            total += j * z[idx]

    return total == k_total


def solution_to_allocation(z, n_buses, max_batt):
    """Convert binary vector to allocation dict."""
    n_choices = max_batt + 1
    alloc = {}
    for i in range(n_buses):
        for j in range(n_choices):
            idx = i * n_choices + j
            if z[idx] == 1:
                alloc[i] = j
    return alloc


# Check all 2^n solutions (warning: exponential!)
best_solution = None
best_cost = float("inf")
valid_solutions = []

print(f"\nSearching {2**n_vars} possible solutions...")
brute_force_start_time = time.time()

for z in product([0, 1], repeat=n_vars):
    cost = evaluate_solution(z, Q)
    is_valid = is_valid_solution(z, n_buses, max_batteries, K)

    if is_valid:
        valid_solutions.append((z, cost))

    if cost < best_cost:
        best_cost = cost
        best_solution = z

# Sort valid solutions by cost
valid_solutions.sort(key=lambda x: x[1])
brute_force_elapsed_time = time.time() - brute_force_start_time
print(f"Brute force completed in {brute_force_elapsed_time:.2f} seconds")
print(f"QAOA completed in {qaoa_elapsed_time:.2f} seconds")

print(f"\nFound {len(valid_solutions)} valid solutions (satisfying all constraints)")
print("\nTop 5 valid solutions:")
print("-" * 70)
for z, cost in valid_solutions[:5]:
    alloc = solution_to_allocation(z, n_buses, max_batteries)
    alloc_str = ", ".join([f"Bus{i}:{v}" for i, v in sorted(alloc.items())])
    total_batt = sum(alloc.values())
    print(f"  [{alloc_str}] Total={total_batt}  QUBO Cost: {cost:8.1f}")

if valid_solutions:
    best_valid = valid_solutions[0]
    best_valid_alloc = solution_to_allocation(best_valid[0], n_buses, max_batteries)
    print(f"\nBest valid solution: {best_valid_alloc}")
    print(f"Best valid QUBO cost: {best_valid[1]}")

qaoa_alloc = interpret_solution(result_x, n_buses, max_batteries, savings, c_install)
print(f"\nQAOA found: {qaoa_alloc} with cost {qubo_objective}")

if valid_solutions and qaoa_alloc == best_valid_alloc:
    print("\n✓ QAOA found the global optimum!")
else:
    print("\n⚠ QAOA found a different solution (may be local optimum)")
