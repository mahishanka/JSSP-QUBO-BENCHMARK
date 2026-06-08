## Methodology

This project benchmarks two QUBO formulations for the classical Job-Shop Scheduling Problem (JSSP). The goal is to compare a standard time-indexed QUBO formulation with a more compact disjunctive QUBO formulation. We use OR-Tools CP-SAT as a classical reference solver and D-Wave's `neal` simulated annealing sampler to solve the QUBO models.

The benchmark is tested on the standard FT06 job-shop scheduling instance. This instance has 6 jobs and 6 machines, with 36 operations in total. The known optimal makespan for FT06 is 55. Instead of directly minimizing the makespan inside the QUBO, we test a list of fixed makespan values `C` and ask whether a feasible schedule exists with makespan at most `C`.

The tested makespan values are:


C = 53, 54, 55, 56, 58, 60
