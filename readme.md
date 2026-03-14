# QUAC 605 - Project: Battery Allocation Optimization for Electric Buses using QAOA
This script demonstrates converting a QUBO matrix to an Ising Hamiltonian and solving it using QAOA.

Problem: Multiple electric buses that distribute energy. We want to install a total of K batteries distributed among the buses. Each bus can have 0, 1, 2, ..., up to max_batteries batteries. The energy saving at each bus depends on the number of batteries installed there.

QUBO Formulation (based on Multi-Knapsack Problem approach):
- Binary variables: x_{i,j} = 1 if bus i gets exactly j batteries
- Objective: Maximize total energy savings (or minimize negative savings)
- Constraints:
  1. Each bus gets exactly one choice: $ \sum_j x_{i,j} = 1$  for all i
  2. Total batteries = K: $\sum_i \sum_j (j * x_{i,j}) = K$

Reference: QUBO Formulations for Multi-Knapsack Problem
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10924197

### Objective function
$x_{i,j} = \text{bus } i \text{ choosing } j \text{ batteries}$
$$
f(x)
=\underbrace{\text{Cost term}}_{\text{maximize savings / minimize cost}}
\;+\;
\underbrace{\lambda_{\text{bus}}\sum_i\left(\sum_j x_{i,j}-1\right)^2}_{\text{“pick exactly one option per bus” constraint}}
\;+\;
\underbrace{\lambda_{\text{total}}\left(\sum_{i,j} j\,x_{i,j}-K\right)^2}_{\text{“total batteries must equal }K\text{” constraint}}
$$
Where:
$$
\textbf{Cost} = \sum_{i,j} \left(-\text{savings}_{i,j} + 50j\right) x_{i,j}.
$$

### QUBO matrix example
Number of buses: 3
Max batteries per bus: 1
Total batteries to distribute: 2
Total binary variables: 6

```
QUBO Matrix Q (shape: 6x6):

        x00     x01     x10     x11     x20     x21
      ------------------------------------------------
x00 |  -500   1000       0       0       0       0
x01 |  1000  -2100       0    1000       0    1000
x10 |     0      0    -500    1000       0       0
x11 |     0   1000    1000   -2030       0    1000
x20 |     0      0       0       0    -500    1000
x21 |     0   1000       0    1000    1000   -2090
```

### How to read a row
Example row: `x01` row (Bus 0 chooses 1 battery)
```
x01: [ 1000  -2100     0  1000     0  1000 ]
        x00   x01   x10   x11   x20   x21
```

* The diagonal **−2100** is its **direct cost** when x01 = 1:
  * **−150** from the objective (we *reward* savings by making them negative in the cost),
  * **+50** installation cost for 1 battery,
  * **−500** from the “exactly one choice for bus 0” penalty term,
  * **−1500** from the “total batteries = 2” penalty term
  * all summed up: −150 + 50 − 500 − 1500 = −2100.
* The **1000** with x00 enforces **“pick exactly one option for bus 0”**:
  * If both x00 and x01 tried to be 1, this pairwise term would add a large penalty.
* The **1000** with x11 and the **1000** with x21 come from the **total battery constraint**:
  * They couple bus 0’s “1 battery” choice with bus 1’s and bus 2’s “1 battery” choices,
  * So that configurations using the wrong total number of batteries get heavily penalized.
