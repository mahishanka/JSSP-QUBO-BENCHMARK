#!/usr/bin/env python3
"""
QI Compact-QUBO Tensor-Network Benchmark for Job-Shop Scheduling
================================================================

This script benchmarks four methods on Job-Shop Scheduling Problem (JSSP)
instances:

    1. Greedy heuristic
    2. Classical local search
    3. OR-Tools CP-SAT
    4. Quantum-inspired Compact-QUBO Tensor-Network LNS

The quantum-inspired method does NOT solve a full QUBO directly.
Instead, it uses a compact-QUBO-style local Hamiltonian inside a
tensor-network / MPS-style beam search.

Main benchmark metrics:
    1. Makespan
    2. Gap %
    3. Runtime

The compact-QUBO idea follows the structure:
    - binary encoded start times,
    - slack variables,
    - machine ordering variables,
    - disjunctive machine constraints.

Here, those ideas are used to build a LOCAL Hamiltonian score for
reordering critical machine blocks, rather than solving the full QUBO.


"""

# ============================================================
# 0. COLAB-SAFE PACKAGE INSTALLATION
# ============================================================

import sys
import subprocess
import importlib.util


def install_if_missing(import_name, pip_name=None):
    """
    Install a package only if it is missing.

    This makes the script easier to run in Google Colab.
    It also avoids reinstalling packages if they already exist.
    """
    if importlib.util.find_spec(import_name) is None:
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )


install_if_missing("job_shop_lib", "job-shop-lib")
install_if_missing("ortools", "ortools")
install_if_missing("pandas", "pandas")
install_if_missing("numpy", "numpy")
install_if_missing("matplotlib", "matplotlib")


# ============================================================
# 1. IMPORTS
# ============================================================

import time
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ortools.sat.python import cp_model
from job_shop_lib.benchmarking import load_benchmark_instance


# Make pandas tables easier to read.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)


# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================

# These are standard benchmark instances.
# For a quick first test, use only ["ft06", "la01", "ft10"].
# For scaling, use the longer list.
INSTANCE_NAMES = ["ft06", "la01", "ft10", "la21", "la24", "la36"]

# Known optimum or best-known makespan values.
# These are used to compute the gap percentage.
KNOWN_OPTIMA = {
    "ft06": 55,
    "la01": 666,
    "ft10": 930,
    "la21": 1046,
    "la24": 935,
    "la36": 1268,
}

# CP-SAT time limit per instance.
# Increase this if you want stronger CP-SAT results.
CP_TIME_LIMIT = 3.0

# Number of rounds for the classical local search.
LOCAL_SEARCH_ROUNDS = 150

# ------------------------------------------------------------
# Quantum-inspired Compact-QUBO TN-LNS settings
# ------------------------------------------------------------

# Size of the machine block we try to reorder.
# Larger block = stronger search but slower.
QI_BLOCK_SIZE = 14

# Effective tensor-network / MPS bond dimension.
# This is the beam width. Larger value = stronger but slower.
QI_BOND_DIM = 96

# Number of large-neighborhood improvement sweeps.
QI_SWEEPS = 6

# Number of candidate critical/bottleneck blocks tested per sweep.
QI_MAX_BLOCKS = 10

# Stop if already very close to the optimum.
QI_STOP_GAP = 0.05

# Small random perturbation used in the beam search.
# This helps break ties and mimic sampling from a quantum-inspired state.
QI_NOISE = 0.08

# Random seed for reproducibility.
QI_SEED = 123

# Weight given to partial schedule completion score inside the beam search.
# Higher value makes the beam more schedule-aware but slightly slower.
COMPLETION_SCORE_WEIGHT = 0.75


# ============================================================
# 3. INSTANCE LOADING
# ============================================================

def first_existing_key(dictionary, keys):
    """
    Return the first value in dictionary whose key appears in keys.

    Different versions of job_shop_lib may use slightly different field names.
    This helper makes the loader more robust.
    """
    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]
    return None


def load_jobs_data(instance_name):
    """
    Load a benchmark instance from job_shop_lib.

    Output format:
        jobs_data[job_id] = [(machine_id, processing_time), ...]

    Example:
        jobs_data[0] = [(2, 1), (0, 3), (1, 6), ...]

    This means job 0 has operations:
        operation 0 uses machine 2 for 1 time unit,
        operation 1 uses machine 0 for 3 time units,
        etc.
    """
    instance = load_benchmark_instance(instance_name)
    data = instance.to_dict()

    durations = first_existing_key(
        data,
        ["duration_matrix", "durations", "processing_times"],
    )

    machines = first_existing_key(
        data,
        ["machines_matrix", "machines", "machine_matrix"],
    )

    if durations is None or machines is None:
        raise KeyError(f"Could not read instance {instance_name}.")

    jobs_data = []

    for machine_row, duration_row in zip(machines, durations):
        job = []

        for machine, processing_time in zip(machine_row, duration_row):
            job.append((int(machine), int(processing_time)))

        jobs_data.append(job)

    # Some datasets use machines numbered 1,2,...,m.
    # We convert them to 0,1,...,m-1.
    all_machines = [machine for job in jobs_data for machine, _ in job]
    n_machines = len(jobs_data[0])

    if min(all_machines) == 1 and max(all_machines) == n_machines:
        jobs_data = [[(machine - 1, p) for machine, p in job] for job in jobs_data]

    metadata = data.get("metadata", {})

    reference = (
        metadata.get("optimum")
        or metadata.get("upper_bound")
        or metadata.get("best_known_solution")
        or metadata.get("best_known")
        or KNOWN_OPTIMA.get(instance_name)
    )

    reference = int(reference) if reference is not None else None

    return jobs_data, reference


# ============================================================
# 4. BASIC JSSP DATA STRUCTURES
# ============================================================

def build_operations(jobs_data):
    """
    Convert jobs_data into operation-level objects.

    Returns:
        ops:
            list of operation dictionaries.

        job_ops:
            job_ops[j] is the list of operation IDs belonging to job j.

        machine_ops:
            machine_ops[m] is the list of operation IDs using machine m.

        precedences:
            list of pairs (a,b), meaning operation a must finish before b starts.
    """
    ops = []
    job_ops = []
    machine_ops = defaultdict(list)
    precedences = []

    op_id = 0

    for job_id, job in enumerate(jobs_data):
        ids_for_this_job = []

        for index_in_job, (machine, processing_time) in enumerate(job):
            operation = {
                "id": op_id,
                "job": job_id,
                "index": index_in_job,
                "machine": machine,
                "p": processing_time,
                "name": f"O{job_id}_{index_in_job}",
            }

            ops.append(operation)
            ids_for_this_job.append(op_id)
            machine_ops[machine].append(op_id)

            op_id += 1

        job_ops.append(ids_for_this_job)

    # Add precedence constraints inside each job.
    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            precedences.append((a, b))

    return ops, job_ops, machine_ops, precedences


def check_schedule(jobs_data, starts):
    """
    Check whether a schedule is feasible.

    A schedule is a dictionary:
        starts[operation_id] = start time

    Feasibility requires:
        1. Every operation has a start time.
        2. Job precedence constraints are satisfied.
        3. No two operations overlap on the same machine.

    Returns:
        feasible: True or False
        makespan: maximum completion time
    """
    if starts is None:
        return False, None

    ops, _, machine_ops, precedences = build_operations(jobs_data)

    expected_ids = {op["id"] for op in ops}

    if set(starts.keys()) != expected_ids:
        return False, None

    makespan = max(starts[a] + ops[a]["p"] for a in starts)

    # Check job precedence.
    for a, b in precedences:
        finish_a = starts[a] + ops[a]["p"]

        if starts[b] < finish_a:
            return False, makespan

    # Check machine non-overlap.
    for _, ids in machine_ops.items():
        intervals = []

        for a in ids:
            start = starts[a]
            end = start + ops[a]["p"]
            intervals.append((start, end, a))

        intervals.sort()

        for (s1, e1, a), (s2, e2, b) in zip(intervals[:-1], intervals[1:]):
            if e1 > s2:
                return False, makespan

    return True, makespan


def machine_orders_from_starts(jobs_data, starts):
    """
    Convert a schedule into a machine order representation.

    For each machine, sort operations by their start time.

    Example:
        machine_orders[0] = [5, 2, 9, 1]
    means machine 0 processes operation 5 first, then 2, then 9, then 1.
    """
    _, _, machine_ops, _ = build_operations(jobs_data)

    machine_orders = {}

    for machine, ids in machine_ops.items():
        machine_orders[machine] = sorted(ids, key=lambda a: starts[a])

    return machine_orders


def schedule_from_machine_orders(jobs_data, machine_orders):
    """
    Decode machine orders into earliest feasible start times.

    Given:
        1. job precedence constraints,
        2. a fixed order of operations on each machine,

    this function computes the earliest-start schedule by longest path
    in the resulting precedence graph.

    If the graph has a cycle, the machine ordering is infeasible.

    Returns:
        starts, makespan, feasible, predecessor_map
    """
    ops, job_ops, _, _ = build_operations(jobs_data)

    n = len(ops)
    adjacency = [[] for _ in range(n)]
    indegree = [0] * n

    # Job precedence arcs.
    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            adjacency[a].append((b, ops[a]["p"]))
            indegree[b] += 1

    # Machine order arcs.
    for _, seq in machine_orders.items():
        for a, b in zip(seq[:-1], seq[1:]):
            adjacency[a].append((b, ops[a]["p"]))
            indegree[b] += 1

    # Topological sorting.
    queue = deque([i for i in range(n) if indegree[i] == 0])
    topo_order = []

    while queue:
        a = queue.popleft()
        topo_order.append(a)

        for b, _ in adjacency[a]:
            indegree[b] -= 1

            if indegree[b] == 0:
                queue.append(b)

    # Cycle means infeasible machine order.
    if len(topo_order) < n:
        return None, None, False, None

    starts = {a: 0 for a in range(n)}
    predecessor = {a: None for a in range(n)}

    # Longest-path dynamic programming.
    for a in topo_order:
        for b, weight in adjacency[a]:
            candidate_start = starts[a] + weight

            if candidate_start > starts[b]:
                starts[b] = candidate_start
                predecessor[b] = a

    makespan = max(starts[a] + ops[a]["p"] for a in range(n))

    return starts, makespan, True, predecessor


def critical_path_from_pred(jobs_data, starts, predecessor):
    """
    Recover one critical path from the predecessor map.

    The critical path ends at the operation with the largest completion time.
    """
    ops, _, _, _ = build_operations(jobs_data)

    last = max(starts.keys(), key=lambda a: starts[a] + ops[a]["p"])

    path = [last]

    while predecessor is not None and predecessor[path[-1]] is not None:
        path.append(predecessor[path[-1]])

    path.reverse()

    return path


def gap_percent(makespan, optimum):
    """
    Compute solution-quality gap percentage.

        gap = 100 * (makespan - optimum) / optimum
    """
    if makespan is None or optimum is None:
        return np.nan

    return 100.0 * (makespan - optimum) / optimum


def keep_better(base_starts, base_C, candidate_starts, candidate_C):
    """
    Keep the better of two schedules.

    This prevents a method from reporting a worse schedule than its warm start.
    """
    if candidate_starts is not None and candidate_C is not None and candidate_C < base_C:
        return candidate_starts, candidate_C

    return base_starts, base_C


# ============================================================
# 5. GREEDY BASELINE
# ============================================================

def greedy_schedule(jobs_data, rule="MWKR"):
    """
    Build a feasible schedule operation by operation.

    At each step, choose one available operation and schedule it as early as possible.

    Rules:
        EST_LPT:
            earliest start time, then longest processing time.

        EST_SPT:
            earliest start time, then shortest processing time.

        MWKR:
            most work remaining.

        MOPNR:
            most operations remaining.

    The best of these rules is used as the Greedy baseline.
    """
    ops, job_ops, _, _ = build_operations(jobs_data)

    n_jobs = len(jobs_data)

    next_op = [0] * n_jobs
    job_ready = [0] * n_jobs
    machine_ready = defaultdict(int)
    starts = {}

    while len(starts) < len(ops):
        candidates = []

        for job_id in range(n_jobs):
            if next_op[job_id] >= len(jobs_data[job_id]):
                continue

            index_in_job = next_op[job_id]
            op_id = job_ops[job_id][index_in_job]
            op = ops[op_id]

            earliest_start = max(
                job_ready[job_id],
                machine_ready[op["machine"]],
            )

            remaining_work = sum(
                p for _, p in jobs_data[job_id][index_in_job:]
            )

            remaining_ops = len(jobs_data[job_id]) - index_in_job

            if rule == "EST_LPT":
                key = (earliest_start, -op["p"])

            elif rule == "EST_SPT":
                key = (earliest_start, op["p"])

            elif rule == "MWKR":
                key = (-remaining_work, earliest_start)

            elif rule == "MOPNR":
                key = (-remaining_ops, earliest_start)

            else:
                key = (earliest_start, -op["p"])

            candidates.append((key, op_id, earliest_start))

        _, chosen_op, start_time = min(candidates, key=lambda item: item[0])

        op = ops[chosen_op]

        starts[chosen_op] = start_time
        job_ready[op["job"]] = start_time + op["p"]
        machine_ready[op["machine"]] = start_time + op["p"]
        next_op[op["job"]] += 1

    feasible, makespan = check_schedule(jobs_data, starts)

    if not feasible:
        raise RuntimeError("Greedy produced an infeasible schedule.")

    return starts, makespan


def best_greedy(jobs_data):
    """
    Run several greedy dispatching rules and keep the best schedule.
    """
    best = None

    for rule in ["EST_LPT", "EST_SPT", "MWKR", "MOPNR"]:
        start_time = time.perf_counter()

        starts, makespan = greedy_schedule(jobs_data, rule=rule)

        runtime = time.perf_counter() - start_time

        candidate = {
            "starts": starts,
            "makespan": makespan,
            "runtime": runtime,
            "rule": rule,
        }

        if best is None or makespan < best["makespan"]:
            best = candidate

    return best


# ============================================================
# 6. CLASSICAL LOCAL SEARCH BASELINE
# ============================================================

def local_search(jobs_data, initial_starts, max_rounds=150):
    """
    Classical local search using machine-order moves.

    Starting from a feasible schedule, convert it into machine orders.
    Then repeatedly try:
        1. adjacent swaps near the critical path,
        2. insertion moves near the critical path.

    If a move improves makespan, accept it.
    Stop when no improving move is found.
    """
    start_time = time.perf_counter()

    machine_orders = machine_orders_from_starts(jobs_data, initial_starts)

    starts, makespan, feasible, predecessor = schedule_from_machine_orders(
        jobs_data,
        machine_orders,
    )

    if not feasible:
        feasible, original_C = check_schedule(jobs_data, initial_starts)
        return initial_starts, original_C, time.perf_counter() - start_time

    for _ in range(max_rounds):
        starts, makespan, feasible, predecessor = schedule_from_machine_orders(
            jobs_data,
            machine_orders,
        )

        critical_set = set(
            critical_path_from_pred(jobs_data, starts, predecessor)
        )

        candidate_moves = []

        # Adjacent swaps involving critical operations.
        for machine, seq in machine_orders.items():
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]

                if a in critical_set or b in critical_set:
                    candidate_moves.append(("swap", machine, i, i + 1))

        # Insertion moves around critical operations.
        for machine, seq in machine_orders.items():
            critical_positions = [
                i for i, op_id in enumerate(seq) if op_id in critical_set
            ]

            for i in critical_positions:
                for j in [i - 3, i - 2, i - 1, i + 1, i + 2, i + 3]:
                    if 0 <= j < len(seq) and i != j:
                        candidate_moves.append(("insert", machine, i, j))

        if not candidate_moves:
            break

        best_move = None

        for move in candidate_moves:
            kind = move[0]
            machine = move[1]

            new_orders = {
                m: list(seq)
                for m, seq in machine_orders.items()
            }

            if kind == "swap":
                _, _, i, j = move
                new_orders[machine][i], new_orders[machine][j] = (
                    new_orders[machine][j],
                    new_orders[machine][i],
                )

            elif kind == "insert":
                _, _, i, j = move
                seq = new_orders[machine]
                item = seq.pop(i)
                seq.insert(j, item)

            new_starts, new_C, ok, _ = schedule_from_machine_orders(
                jobs_data,
                new_orders,
            )

            if ok and new_C < makespan:
                if best_move is None or new_C < best_move["makespan"]:
                    best_move = {
                        "orders": new_orders,
                        "starts": new_starts,
                        "makespan": new_C,
                    }

        if best_move is None:
            break

        machine_orders = best_move["orders"]
        starts = best_move["starts"]
        makespan = best_move["makespan"]

    runtime = time.perf_counter() - start_time

    return starts, makespan, runtime


# ============================================================
# 7. CP-SAT BASELINE
# ============================================================

def cpsat_solve(jobs_data, time_limit=3.0, hint_starts=None):
    """
    Solve the JSSP using OR-Tools CP-SAT.

    CP-SAT is a strong classical constraint-programming solver.

    We include it as a strong classical baseline.
    The time limit is intentionally short so that comparison is fair
    for a quick experimental benchmark.
    """
    start_time = time.perf_counter()

    ops, job_ops, machine_ops, _ = build_operations(jobs_data)

    horizon = sum(op["p"] for op in ops)

    model = cp_model.CpModel()

    start_vars = {}
    end_vars = {}
    intervals_by_machine = defaultdict(list)

    for op in ops:
        op_id = op["id"]

        start = model.NewIntVar(0, horizon, f"s_{op_id}")
        end = model.NewIntVar(0, horizon, f"e_{op_id}")
        interval = model.NewIntervalVar(start, op["p"], end, f"I_{op_id}")

        start_vars[op_id] = start
        end_vars[op_id] = end
        intervals_by_machine[op["machine"]].append(interval)

    # Job precedence constraints.
    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            model.Add(start_vars[b] >= end_vars[a])

    # Machine non-overlap constraints.
    for _, intervals in intervals_by_machine.items():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")

    for op in ops:
        model.Add(makespan >= end_vars[op["id"]])

    model.Minimize(makespan)

    # Warm-start CP-SAT with the greedy schedule.
    if hint_starts is not None:
        for op_id in start_vars:
            model.AddHint(start_vars[op_id], int(hint_starts[op_id]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    runtime = time.perf_counter() - start_time

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None, None, runtime

    starts = {
        op_id: solver.Value(start_vars[op_id])
        for op_id in start_vars
    }

    feasible, C = check_schedule(jobs_data, starts)

    if not feasible:
        return None, None, runtime

    return starts, C, runtime


# ============================================================
# 8. QUANTUM-INSPIRED COMPACT-QUBO TN-LNS METHOD
# ============================================================

def compact_qubo_local_hamiltonian(jobs_data, starts, block, critical_set):
    """
    Construct a local Compact-QUBO-inspired Hamiltonian.

    This is the main quantum-inspired part.

    Instead of building a full QUBO for the whole instance, we select a
    critical machine block and build a local energy model on that block.

    The local Hamiltonian contains terms inspired by compact disjunctive QUBO:
        1. precedence penalty:
            bad if the local order violates job precedence;

        2. critical-path interaction:
            operations on the critical path receive stronger attention;

        3. low-slack interaction:
            operations with little slack are important;

        4. processing-time interaction:
            long operations are important in scheduling.

    The output H is a k x k matrix where:
        H[i,j] = energy contribution if operation i is placed before operation j.
    """
    ops, job_ops, _, _ = build_operations(jobs_data)

    idx = {op_id: i for i, op_id in enumerate(block)}

    H = np.zeros((len(block), len(block)), dtype=float)

    # Precedence pairs inside the selected block.
    precedence_pairs = set()

    for ids in job_ops:
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a in idx and b in idx:
                    precedence_pairs.add((a, b))

    current_C = max(starts[a] + ops[a]["p"] for a in starts)

    finish = {
        a: starts[a] + ops[a]["p"]
        for a in starts
    }

    slack = {
        a: max(0, current_C - finish[a])
        for a in starts
    }

    for before in block:
        for after in block:
            if before == after:
                continue

            i = idx[before]
            j = idx[after]

            value = 0.0

            # If after must be before before, then before -> after is bad.
            if (after, before) in precedence_pairs:
                value += 100.0

            # Critical-path interaction.
            if before in critical_set or after in critical_set:
                value += 8.0

            # Low-slack interaction.
            value += 0.04 * (current_C - min(slack[before], slack[after]))

            # Processing-time interaction.
            value += 0.08 * (ops[before]["p"] + ops[after]["p"])

            H[i, j] = value

    return H


def candidate_blocks_for_qi(jobs_data, current_starts, rng):
    """
    Select promising machine blocks for the QI/TN improvement step.

    We use two ideas:
        1. blocks centered around operations on the critical path;
        2. blocks from highly loaded bottleneck machines.

    These are the places where changing the machine order is most likely
    to reduce makespan.
    """
    ops, _, _, _ = build_operations(jobs_data)

    machine_orders = machine_orders_from_starts(jobs_data, current_starts)

    starts, _, ok, predecessor = schedule_from_machine_orders(
        jobs_data,
        machine_orders,
    )

    if not ok:
        return []

    critical_path = critical_path_from_pred(jobs_data, starts, predecessor)
    critical_set = set(critical_path)

    candidates = []

    def add_block(machine, block):
        block = list(block)

        if len(block) < 4:
            return

        critical_density = sum(
            1 for op_id in block if op_id in critical_set
        ) / len(block)

        load = sum(ops[op_id]["p"] for op_id in block)

        # Score blocks by how critical and loaded they are.
        score = 120.0 * critical_density + 0.25 * load + rng.random()

        candidates.append({
            "score": score,
            "machine": machine,
            "block": block,
        })

    # Critical-path-centered blocks.
    for op_id in critical_path:
        machine = ops[op_id]["machine"]
        seq = machine_orders[machine]
        pos = seq.index(op_id)

        half = QI_BLOCK_SIZE // 2

        lo = max(0, pos - half)
        hi = min(len(seq), lo + QI_BLOCK_SIZE)
        lo = max(0, hi - QI_BLOCK_SIZE)

        add_block(machine, seq[lo:hi])

    # Bottleneck-machine blocks.
    machine_loads = {
        machine: sum(ops[op_id]["p"] for op_id in seq)
        for machine, seq in machine_orders.items()
    }

    for machine in sorted(machine_loads, key=machine_loads.get, reverse=True)[:5]:
        seq = machine_orders[machine]

        if len(seq) < 4:
            continue

        step = max(1, QI_BLOCK_SIZE // 3)

        for lo in range(0, max(1, len(seq) - QI_BLOCK_SIZE + 1), step):
            hi = min(len(seq), lo + QI_BLOCK_SIZE)
            add_block(machine, seq[lo:hi])

    # Remove duplicate blocks.
    seen = set()
    unique = []

    for candidate in candidates:
        key = (candidate["machine"], tuple(candidate["block"]))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    unique.sort(key=lambda item: item["score"], reverse=True)

    return unique[:QI_MAX_BLOCKS]


def evaluate_block_order(jobs_data, machine_orders, machine, block, permutation):
    """
    Replace the selected block on one machine by a new permutation.

    Then decode the full schedule and compute the makespan.
    """
    seq = machine_orders[machine]

    positions = [seq.index(op_id) for op_id in block]

    lo = min(positions)
    hi = max(positions) + 1

    new_orders = {
        m: list(s)
        for m, s in machine_orders.items()
    }

    new_orders[machine] = seq[:lo] + list(permutation) + seq[hi:]

    starts, C, feasible, _ = schedule_from_machine_orders(
        jobs_data,
        new_orders,
    )

    return starts, C, feasible


def qi_mps_beam_search_block(jobs_data, current_starts, machine, block, rng):
    """
    Quantum-inspired MPS/TN beam decoder.

    Goal:
        Find a better ordering of a selected machine block.

    Why this is tensor-network inspired:
        A full ordering of k operations has k! possibilities.
        Instead of enumerating all k! possibilities, we build the sequence
        one layer at a time and keep only QI_BOND_DIM partial states.

        This is analogous to an MPS keeping a fixed bond dimension chi.

    Effective bond dimension:
        chi = QI_BOND_DIM

    The score used in the beam combines:
        1. compact-QUBO local Hamiltonian energy,
        2. schedule-informed completion score,
        3. small noise for diversity.
    """
    ops, job_ops, _, _ = build_operations(jobs_data)

    machine_orders = machine_orders_from_starts(jobs_data, current_starts)

    starts0, incumbent_C, ok, predecessor = schedule_from_machine_orders(
        jobs_data,
        machine_orders,
    )

    if not ok:
        return None

    block = list(block)
    block_set = set(block)
    block_index = {op_id: i for i, op_id in enumerate(block)}

    # Local precedence restrictions inside the block.
    predecessors = {op_id: set() for op_id in block}

    for ids in job_ops:
        seen = []

        for op_id in ids:
            if op_id in block_set:
                predecessors[op_id].update(x for x in seen if x in block_set)
                seen.append(op_id)

    critical_set = set(
        critical_path_from_pred(jobs_data, starts0, predecessor)
    )

    H = compact_qubo_local_hamiltonian(
        jobs_data,
        current_starts,
        block,
        critical_set,
    )

    current_position = {
        op_id: i
        for i, op_id in enumerate(machine_orders[machine])
    }

    completion_cache = {}

    def complete_partial_sequence(seq, remaining):
        """
        Complete a partial sequence by appending remaining operations in
        their incumbent order.

        This gives a cheap way to estimate the makespan of a partial beam state.
        """
        remaining_sorted = tuple(
            sorted(remaining, key=lambda op_id: current_position[op_id])
        )

        return tuple(seq) + remaining_sorted

    def completion_makespan(seq, remaining):
        """
        Estimate how good a partial sequence is by completing it and decoding
        the full schedule.
        """
        key = (seq, tuple(sorted(remaining)))

        if key in completion_cache:
            return completion_cache[key]

        full_sequence = complete_partial_sequence(seq, remaining)

        _, C_new, feasible = evaluate_block_order(
            jobs_data,
            machine_orders,
            machine,
            block,
            full_sequence,
        )

        if not feasible:
            C_new = incumbent_C + 10**6

        completion_cache[key] = C_new

        return C_new

    def partial_hamiltonian_energy(seq):
        """
        Compact-QUBO-inspired energy of a partial sequence.
        """
        energy = 0.0

        for i, op_id in enumerate(seq):
            # Encourage critical operations earlier.
            if op_id in critical_set:
                energy -= 7.0 / (i + 1)

            # Mild long-processing-time bias.
            energy -= 0.08 * ops[op_id]["p"] / (i + 1)

            # Do not destroy the current order too aggressively.
            energy += 0.01 * abs(i - current_position[op_id])

            op_i = block_index[op_id]

            # Pairwise compact-QUBO Hamiltonian contribution.
            for j in range(i):
                previous_op = seq[j]
                op_j = block_index[previous_op]
                energy += H[op_j, op_i]

        return energy

    # Beam state:
    #     (score, partial_sequence, remaining_operations)
    beam = [(0.0, tuple(), frozenset(block))]

    for _ in range(len(block)):
        new_beam = []

        for _, seq, remaining in beam:
            placed = set(seq)

            for op_id in remaining:
                # Enforce local job precedence.
                if not predecessors[op_id].issubset(placed):
                    continue

                new_seq = seq + (op_id,)
                new_remaining = frozenset(x for x in remaining if x != op_id)

                h_score = partial_hamiltonian_energy(new_seq)

                # Schedule-aware completion score.
                c_score = completion_makespan(new_seq, new_remaining)

                score = h_score + COMPLETION_SCORE_WEIGHT * (c_score - incumbent_C)

                # Small stochastic perturbation.
                score += QI_NOISE * rng.random()

                new_beam.append((score, new_seq, new_remaining))

        if not new_beam:
            return None

        # Keep only the best QI_BOND_DIM partial states.
        new_beam.sort(key=lambda item: item[0])
        beam = new_beam[:QI_BOND_DIM]

    # Evaluate the final full sequences exactly.
    best = None

    for _, seq, _ in beam:
        new_starts, new_C, feasible = evaluate_block_order(
            jobs_data,
            machine_orders,
            machine,
            block,
            seq,
        )

        if feasible and new_C < incumbent_C:
            if best is None or new_C < best["makespan"]:
                best = {
                    "starts": new_starts,
                    "makespan": new_C,
                }

    return best


def qi_compact_qubo_tn_lns(jobs_data, initial_starts, optimum, seed=0):
    """
    Main quantum-inspired Compact-QUBO TN-LNS method.

    Workflow:
        1. Start from an incumbent schedule.
        2. Find critical/bottleneck machine blocks.
        3. Build compact-QUBO local Hamiltonian for each block.
        4. Use MPS/TN beam search to find a better local ordering.
        5. Accept the best feasible improvement.
        6. Repeat for several sweeps.

    This is a hybrid method:
        Local Search gives a strong incumbent.
        QI Compact-QUBO TN-LNS tries to improve it using larger blocks.
    """
    rng = random.Random(seed)

    start_time = time.perf_counter()

    current_starts = dict(initial_starts)

    feasible, current_C = check_schedule(jobs_data, current_starts)

    if not feasible:
        return None, None, 0.0, 0

    completed_sweeps = 0

    for _ in range(QI_SWEEPS):
        current_gap = gap_percent(current_C, optimum)

        if not np.isnan(current_gap) and current_gap <= QI_STOP_GAP:
            break

        candidate_blocks = candidate_blocks_for_qi(
            jobs_data,
            current_starts,
            rng,
        )

        if not candidate_blocks:
            break

        best_move = None

        for block_info in candidate_blocks:
            candidate = qi_mps_beam_search_block(
                jobs_data,
                current_starts,
                block_info["machine"],
                block_info["block"],
                rng,
            )

            if candidate is not None:
                if best_move is None or candidate["makespan"] < best_move["makespan"]:
                    best_move = candidate

        completed_sweeps += 1

        if best_move is None:
            break

        current_starts = best_move["starts"]
        current_C = best_move["makespan"]

    runtime = time.perf_counter() - start_time

    feasible, final_C = check_schedule(jobs_data, current_starts)

    if not feasible:
        return None, None, runtime, completed_sweeps

    return current_starts, final_C, runtime, completed_sweeps


# ============================================================
# 9. BENCHMARK ONE INSTANCE
# ============================================================

def benchmark_instance(instance_name):
    """
    Run all four methods on one JSSP instance.

    Methods:
        1. Greedy
        2. Local Search
        3. CP-SAT
        4. QI Compact-QUBO TN-LNS

    Metrics:
        1. Makespan
        2. Gap %
        3. Runtime
    """
    print(f"\nRunning {instance_name}...")

    jobs_data, optimum = load_jobs_data(instance_name)

    rows = []

    # --------------------------------------------------------
    # Method 1: Greedy
    # --------------------------------------------------------
    greedy = best_greedy(jobs_data)

    feasible, C_greedy = check_schedule(jobs_data, greedy["starts"])

    rows.append({
        "Instance": instance_name,
        "Method": "Greedy",
        "Makespan": C_greedy,
        "Gap_%": gap_percent(C_greedy, optimum),
        "Runtime_s": greedy["runtime"],
    })

    # --------------------------------------------------------
    # Method 2: Classical Local Search
    # --------------------------------------------------------
    ls_starts, C_ls, ls_runtime = local_search(
        jobs_data,
        greedy["starts"],
        max_rounds=LOCAL_SEARCH_ROUNDS,
    )

    feasible, C_ls = check_schedule(jobs_data, ls_starts)

    rows.append({
        "Instance": instance_name,
        "Method": "Local Search",
        "Makespan": C_ls,
        "Gap_%": gap_percent(C_ls, optimum),
        "Runtime_s": greedy["runtime"] + ls_runtime,
    })

    # --------------------------------------------------------
    # Method 3: CP-SAT
    # --------------------------------------------------------
    cp_starts, C_cp, cp_runtime = cpsat_solve(
        jobs_data,
        time_limit=CP_TIME_LIMIT,
        hint_starts=greedy["starts"],
    )

    cp_starts, C_cp = keep_better(
        greedy["starts"],
        C_greedy,
        cp_starts,
        C_cp,
    )

    feasible, C_cp = check_schedule(jobs_data, cp_starts)

    rows.append({
        "Instance": instance_name,
        "Method": "CP-SAT",
        "Makespan": C_cp,
        "Gap_%": gap_percent(C_cp, optimum),
        "Runtime_s": greedy["runtime"] + cp_runtime,
    })

    # --------------------------------------------------------
    # Method 4: QI Compact-QUBO TN-LNS
    # --------------------------------------------------------
    # We warm-start from Local Search because the QI method is an
    # improvement layer, not a standalone solver.
    qi_starts, C_qi, qi_runtime, qi_sweeps = qi_compact_qubo_tn_lns(
        jobs_data,
        ls_starts,
        optimum,
        seed=QI_SEED,
    )

    feasible, C_qi = check_schedule(jobs_data, qi_starts)

    rows.append({
        "Instance": instance_name,
        "Method": "QI Compact-QUBO TN-LNS",
        "Makespan": C_qi,
        "Gap_%": gap_percent(C_qi, optimum),
        "Runtime_s": greedy["runtime"] + ls_runtime + qi_runtime,
    })

    return pd.DataFrame(rows)


# ============================================================
# 10. RUN FULL BENCHMARK
# ============================================================

all_results = []

for instance_name in INSTANCE_NAMES:
    try:
        result = benchmark_instance(instance_name)
        all_results.append(result)

    except Exception as error:
        print(f"Skipping {instance_name} because of error: {error}")

results = pd.concat(all_results, ignore_index=True)

for column in ["Makespan", "Gap_%", "Runtime_s"]:
    results[column] = pd.to_numeric(results[column], errors="coerce").round(4)

print("\nMAIN BENCHMARK TABLE")
print(results)


# ============================================================
# 11. METRIC TABLES
# ============================================================

makespan_table = results.pivot_table(
    index="Instance",
    columns="Method",
    values="Makespan",
    aggfunc="first",
).reset_index()

gap_table = results.pivot_table(
    index="Instance",
    columns="Method",
    values="Gap_%",
    aggfunc="first",
).reset_index()

runtime_table = results.pivot_table(
    index="Instance",
    columns="Method",
    values="Runtime_s",
    aggfunc="first",
).reset_index()

print("\nMETRIC 1: MAKESPAN")
print(makespan_table)

print("\nMETRIC 2: GAP %")
print(gap_table)

print("\nMETRIC 3: RUNTIME")
print(runtime_table)


# ============================================================
# 12. ONLY THREE GRAPHS
# ============================================================

METHOD_ORDER = [
    "Greedy",
    "Local Search",
    "CP-SAT",
    "QI Compact-QUBO TN-LNS",
]

INSTANCE_ORDER = [
    instance for instance in INSTANCE_NAMES
    if instance in results["Instance"].unique()
]


def plot_metric(metric, title, ylabel, logy=False):
    """
    Plot one metric as a grouped bar chart.
    """
    pivot = results.pivot_table(
        index="Instance",
        columns="Method",
        values=metric,
        aggfunc="first",
    )

    pivot = pivot.reindex(INSTANCE_ORDER)

    available_methods = [
        method for method in METHOD_ORDER
        if method in pivot.columns
    ]

    pivot = pivot[available_methods]

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 5),
        width=0.82,
    )

    ax.set_title(title)
    ax.set_xlabel("Instance")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    if logy:
        ax.set_yscale("log")

    plt.xticks(rotation=0)

    plt.legend(
        title="Method",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


plot_metric(
    metric="Makespan",
    title="Metric 1: Makespan",
    ylabel="Makespan",
    logy=False,
)

plot_metric(
    metric="Gap_%",
    title="Metric 2: Solution Quality Gap %",
    ylabel="Gap %",
    logy=False,
)

plot_metric(
    metric="Runtime_s",
    title="Metric 3: Runtime",
    ylabel="Runtime (seconds)",
    logy=True,
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results.to_csv("qi_compactqubo_tn_jssp_benchmark_results.csv", index=False)
makespan_table.to_csv("metric_1_makespan_table.csv", index=False)
gap_table.to_csv("metric_2_gap_table.csv", index=False)
runtime_table.to_csv("metric_3_runtime_table.csv", index=False)

print("\nSaved files:")
print("qi_compactqubo_tn_jssp_benchmark_results.csv")
print("metric_1_makespan_table.csv")
print("metric_2_gap_table.csv")
print("metric_3_runtime_table.csv")
