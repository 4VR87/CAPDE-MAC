# CAPDE-MAC
CAPDE-MAC is a standard-compatible IEEE 802.15.6 control overlay that combines deadline-risk-aware scheduling, adaptive access partitioning, emergency overrides, and ageing-based starvation prevention. In simulation at 1.8× load, it achieved 142.22 ms mean delay, 300 ms P95 delay, 1.03% deadline misses, and 93.6 μJ per delivered packet.

# CAPDE-MAC code and dataset
This contains the Python 3 code and datasets used for the common-scenario comparison of:
1. CAPDE-MAC
2. ADT-MAC
3. MDP-HYMAC
4. Static reference
5. Priority CSMA/CA
6. CAPDE-MAC without prediction
7. CAPDE-MAC without emergency pre-emption

## Experiment matrix
- 7 protocol configurations
- 4 offered-load multipliers: 0.6, 1.0, 1.4, 1.8
- 30 deterministic seeds per protocol/load combination
- total = 7 × 4 × 30 = 840 simulation runs
- 12 heterogeneous WBAN sensor nodes
- 180 s per run
- 100 ms frame / beacon interval
- 5 service opportunities per frame
