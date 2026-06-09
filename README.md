## Methodology

This project benchmarks two QUBO formulations for the classical Job-Shop Scheduling Problem (JSSP). The goal is to compare a standard time-indexed QUBO formulation with a more compact disjunctive QUBO formulation. We use OR-Tools CP-SAT as a classical reference solver and D-Wave's `neal` simulated annealing sampler to solve the QUBO models.

The benchmark allows for testing using a variety of standard instances against their known optimum makespan; see https://scheduleopt.github.io/benchmarks/jsplib/. 

Instead of directly minimizing the makespan inside the QUBO, we test a list of fixed makespan values `C` and ask whether a feasible schedule exists with makespan at most `C`.
