#!/usr/bin/env python3
"""
QI Compact-QUBO Tensor-Network Benchmark for Job-Shop Scheduling

Methods:
    1. Greedy
    2. QUBO
    3. CP-SAT
    4. QI Compact-QUBO TN-LNS

Local Search has been removed and replaced by QUBO.
"""

# ============================================================
# 0. COLAB-SAFE PACKAGE INSTALLATION
# ============================================================

import sys
import subprocess
import importlib.util


def install_if_missing(import_name, pip_name=None):
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


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)


# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================

INSTANCE_NAMES = ["ft06", "la01", "ft10", "la21", "la24", "la36"]

KNOWN_OPTIMA = {
    "ft06": 55,
    "la01": 666,
    "ft10": 930,
    "la21": 1046,
    "la24": 935,
    "la36": 1268,
}

CP_TIME_LIMIT = 3.0

# QUBO baseline settings
QUBO_BLOCK_SIZE = 12
QUBO_SWEEPS = 6
QUBO_MAX_BLOCKS = 10
QUBO_SA_STEPS = 600
QUBO_START_TEMP = 6.0
QUBO_END_TEMP = 0.05
QUBO_STOP_GAP = 0.05

# Tensor-network method settings
QI_BLOCK_SIZE = 14
QI_BOND_DIM = 96
QI_SWEEPS = 6
QI_MAX_BLOCKS = 10
QI_STOP_GAP = 0.05
QI_NOISE = 0.08
QI_SEED = 123
COMPLETION_SCORE_WEIGHT = 0.75


# ============================================================
# 3. INSTANCE LOADING
# ============================================================

def first_existing_key(dictionary, keys):
    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]
    return None


def load_jobs_data(instance_name):
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
# 4. BASIC JSSP STRUCTURES
# ============================================================

def build_operations(jobs_data):
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

    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            precedences.append((a, b))

    return ops, job_ops, machine_ops, precedences


def check_schedule(jobs_data, starts):
    if starts is None:
        return False, None

    ops, _, machine_ops, precedences = build_operations(jobs_data)

    expected_ids = {op["id"] for op in ops}

    if set(starts.keys()) != expected_ids:
        return False, None

    makespan = max(starts[a] + ops[a]["p"] for a in starts)

    for a, b in precedences:
        if starts[b] < starts[a] + ops[a]["p"]:
            return False, makespan

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
    _, _, machine_ops, _ = build_operations(jobs_data)

    machine_orders = {}

    for machine, ids in machine_ops.items():
        machine_orders[machine] = sorted(ids, key=lambda a: starts[a])

    return machine_orders


def schedule_from_machine_orders(jobs_data, machine_orders):
    ops, job_ops, _, _ = build_operations(jobs_data)

    n = len(ops)
    adjacency = [[] for _ in range(n)]
    indegree = [0] * n

    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            adjacency[a].append((b, ops[a]["p"]))
            indegree[b] += 1

    for _, seq in machine_orders.items():
        for a, b in zip(seq[:-1], seq[1:]):
            adjacency[a].append((b, ops[a]["p"]))
            indegree[b] += 1

    queue = deque([i for i in range(n) if indegree[i] == 0])
    topo_order = []

    while queue:
        a = queue.popleft()
        topo_order.append(a)

        for b, _ in adjacency[a]:
            indegree[b] -= 1

            if indegree[b] == 0:
                queue.append(b)

    if len(topo_order) < n:
        return None, None, False, None

    starts = {a: 0 for a in range(n)}
    predecessor = {a: None for a in range(n)}

    for a in topo_order:
        for b, weight in adjacency[a]:
            candidate_start = starts[a] + weight

            if candidate_start > starts[b]:
                starts[b] = candidate_start
                predecessor[b] = a

    makespan = max(starts[a] + ops[a]["p"] for a in range(n))

    return starts, makespan, True, predecessor


def critical_path_from_pred(jobs_data, starts, predecessor):
    ops, _, _, _ = build_operations(jobs_data)

    last = max(starts.keys(), key=lambda a: starts[a] + ops[a]["p"])

    path = [last]

    while predecessor is not None and predecessor[path[-1]] is not None:
        path.append(predecessor[path[-1]])

    path.reverse()

    return path


def gap_percent(makespan, optimum):
    if makespan is None or optimum is None:
        return np.nan

    return 100.0 * (makespan - optimum) / optimum


def keep_better(base_starts, base_C, candidate_starts, candidate_C):
    if candidate_starts is not None and candidate_C is not None and candidate_C < base_C:
        return candidate_starts, candidate_C

    return base_starts, base_C


# ============================================================
# 5. GREEDY BASELINE
# ============================================================

def greedy_schedule(jobs_data, rule="MWKR"):
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
# 6. COMMON BLOCK EVALUATION
# ============================================================

def evaluate_block_order(jobs_data, machine_orders, machine, block, permutation):
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


def compact_qubo_local_hamiltonian(jobs_data, starts, block, critical_set):
    ops, job_ops, _, _ = build_operations(jobs_data)

    idx = {op_id: i for i, op_id in enumerate(block)}

    H = np.zeros((len(block), len(block)), dtype=float)

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

            if (after, before) in precedence_pairs:
                value += 100.0

            if before in critical_set or after in critical_set:
                value += 8.0

            value += 0.04 * (current_C - min(slack[before], slack[after]))

            value += 0.08 * (ops[before]["p"] + ops[after]["p"])

            H[i, j] = value

    return H


# ============================================================
# 7. STANDALONE QUBO BASELINE
# ============================================================

def candidate_blocks_for_qubo(jobs_data, current_starts, rng):
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

        score = 120.0 * critical_density + 0.25 * load + rng.random()

        candidates.append({
            "score": score,
            "machine": machine,
            "block": block,
        })

    for op_id in critical_path:
        machine = ops[op_id]["machine"]
        seq = machine_orders[machine]
        pos = seq.index(op_id)

        half = QUBO_BLOCK_SIZE // 2

        lo = max(0, pos - half)
        hi = min(len(seq), lo + QUBO_BLOCK_SIZE)
        lo = max(0, hi - QUBO_BLOCK_SIZE)

        add_block(machine, seq[lo:hi])

    machine_loads = {
        machine: sum(ops[op_id]["p"] for op_id in seq)
        for machine, seq in machine_orders.items()
    }

    for machine in sorted(machine_loads, key=machine_loads.get, reverse=True)[:5]:
        seq = machine_orders[machine]

        if len(seq) < 4:
            continue

        step = max(1, QUBO_BLOCK_SIZE // 3)

        for lo in range(0, max(1, len(seq) - QUBO_BLOCK_SIZE + 1), step):
            hi = min(len(seq), lo + QUBO_BLOCK_SIZE)
            add_block(machine, seq[lo:hi])

    seen = set()
    unique = []

    for candidate in candidates:
        key = (candidate["machine"], tuple(candidate["block"]))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    unique.sort(key=lambda item: item["score"], reverse=True)

    return unique[:QUBO_MAX_BLOCKS]


def local_qubo_energy_for_sequence(
    jobs_data,
    starts,
    block,
    sequence,
    critical_set,
    incumbent_position,
):
    ops, job_ops, _, _ = build_operations(jobs_data)

    k = len(block)
    block_set = set(block)
    pos = {op_id: i for i, op_id in enumerate(sequence)}

    current_C = max(starts[a] + ops[a]["p"] for a in starts)

    finish = {
        a: starts[a] + ops[a]["p"]
        for a in starts
    }

    slack = {
        a: max(0, current_C - finish[a])
        for a in starts
    }

    energy = 0.0

    for op_id in sequence:
        t = pos[op_id]
        op = ops[op_id]

        critical_bonus = 1.0 if op_id in critical_set else 0.0
        low_slack_score = current_C - slack[op_id]
        long_operation_score = op["p"]

        priority = (
            9.0 * critical_bonus
            + 0.06 * low_slack_score
            + 0.12 * long_operation_score
        )

        energy -= priority * (k - t) / max(1, k)

        energy += 0.03 * abs(t - incumbent_position[op_id])

    for ids in job_ops:
        local_ids = [op_id for op_id in ids if op_id in block_set]

        for i, a in enumerate(local_ids):
            for b in local_ids[i + 1:]:
                if pos[a] > pos[b]:
                    energy += 10000.0

    H = compact_qubo_local_hamiltonian(
        jobs_data,
        starts,
        block,
        critical_set,
    )

    idx = {op_id: i for i, op_id in enumerate(block)}

    for i in range(k):
        for j in range(i + 1, k):
            before = sequence[i]
            after = sequence[j]
            energy += H[idx[before], idx[after]]

    return energy


def solve_local_qubo_by_annealing(
    jobs_data,
    starts,
    machine,
    block,
    critical_set,
    rng,
):
    incumbent_sequence = tuple(block)

    incumbent_position = {
        op_id: i
        for i, op_id in enumerate(incumbent_sequence)
    }

    current_sequence = incumbent_sequence
    current_energy = local_qubo_energy_for_sequence(
        jobs_data,
        starts,
        block,
        current_sequence,
        critical_set,
        incumbent_position,
    )

    best_sequence = current_sequence
    best_energy = current_energy

    k = len(block)

    if k < 2:
        return list(best_sequence)

    for step in range(QUBO_SA_STEPS):
        alpha = step / max(1, QUBO_SA_STEPS - 1)
        temperature = QUBO_START_TEMP * (
            QUBO_END_TEMP / QUBO_START_TEMP
        ) ** alpha

        candidate = list(current_sequence)

        if rng.random() < 0.50:
            i, j = rng.sample(range(k), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            i, j = rng.sample(range(k), 2)
            item = candidate.pop(i)
            candidate.insert(j, item)

        candidate = tuple(candidate)

        candidate_energy = local_qubo_energy_for_sequence(
            jobs_data,
            starts,
            block,
            candidate,
            critical_set,
            incumbent_position,
        )

        delta = candidate_energy - current_energy

        if delta <= 0:
            accept = True
        else:
            accept_probability = np.exp(-delta / max(temperature, 1e-9))
            accept = rng.random() < accept_probability

        if accept:
            current_sequence = candidate
            current_energy = candidate_energy

            if current_energy < best_energy:
                best_sequence = current_sequence
                best_energy = current_energy

    return list(best_sequence)


def compact_qubo_baseline(jobs_data, initial_starts, optimum, seed=0):
    rng = random.Random(seed)

    start_time = time.perf_counter()

    current_starts = dict(initial_starts)

    feasible, current_C = check_schedule(jobs_data, current_starts)

    if not feasible:
        return None, None, 0.0, 0

    completed_sweeps = 0

    for _ in range(QUBO_SWEEPS):
        current_gap = gap_percent(current_C, optimum)

        if not np.isnan(current_gap) and current_gap <= QUBO_STOP_GAP:
            break

        machine_orders = machine_orders_from_starts(jobs_data, current_starts)

        starts0, incumbent_C, ok, predecessor = schedule_from_machine_orders(
            jobs_data,
            machine_orders,
        )

        if not ok:
            break

        critical_set = set(
            critical_path_from_pred(jobs_data, starts0, predecessor)
        )

        candidate_blocks = candidate_blocks_for_qubo(
            jobs_data,
            current_starts,
            rng,
        )

        if not candidate_blocks:
            break

        best_move = None

        for block_info in candidate_blocks:
            machine = block_info["machine"]
            block = block_info["block"]

            candidate_sequence = solve_local_qubo_by_annealing(
                jobs_data,
                current_starts,
                machine,
                block,
                critical_set,
                rng,
            )

            candidate_starts, candidate_C, feasible = evaluate_block_order(
                jobs_data,
                machine_orders,
                machine,
                block,
                candidate_sequence,
            )

            if feasible and candidate_C < incumbent_C:
                if best_move is None or candidate_C < best_move["makespan"]:
                    best_move = {
                        "starts": candidate_starts,
                        "makespan": candidate_C,
                    }

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
# 8. CP-SAT BASELINE
# ============================================================

def cpsat_solve(jobs_data, time_limit=3.0, hint_starts=None):
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

    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            model.Add(start_vars[b] >= end_vars[a])

    for _, intervals in intervals_by_machine.items():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")

    for op in ops:
        model.Add(makespan >= end_vars[op["id"]])

    model.Minimize(makespan)

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
# 9. QI COMPACT-QUBO TENSOR-NETWORK METHOD
# ============================================================

def candidate_blocks_for_qi(jobs_data, current_starts, rng):
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

        score = 120.0 * critical_density + 0.25 * load + rng.random()

        candidates.append({
            "score": score,
            "machine": machine,
            "block": block,
        })

    for op_id in critical_path:
        machine = ops[op_id]["machine"]
        seq = machine_orders[machine]
        pos = seq.index(op_id)

        half = QI_BLOCK_SIZE // 2

        lo = max(0, pos - half)
        hi = min(len(seq), lo + QI_BLOCK_SIZE)
        lo = max(0, hi - QI_BLOCK_SIZE)

        add_block(machine, seq[lo:hi])

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

    seen = set()
    unique = []

    for candidate in candidates:
        key = (candidate["machine"], tuple(candidate["block"]))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    unique.sort(key=lambda item: item["score"], reverse=True)

    return unique[:QI_MAX_BLOCKS]


def qi_mps_beam_search_block(jobs_data, current_starts, machine, block, rng):
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
        remaining_sorted = tuple(
            sorted(remaining, key=lambda op_id: current_position[op_id])
        )

        return tuple(seq) + remaining_sorted

    def completion_makespan(seq, remaining):
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
        energy = 0.0

        for i, op_id in enumerate(seq):
            if op_id in critical_set:
                energy -= 7.0 / (i + 1)

            energy -= 0.08 * ops[op_id]["p"] / (i + 1)

            energy += 0.01 * abs(i - current_position[op_id])

            op_i = block_index[op_id]

            for j in range(i):
                previous_op = seq[j]
                op_j = block_index[previous_op]
                energy += H[op_j, op_i]

        return energy

    beam = [(0.0, tuple(), frozenset(block))]

    for _ in range(len(block)):
        new_beam = []

        for _, seq, remaining in beam:
            placed = set(seq)

            for op_id in remaining:
                if not predecessors[op_id].issubset(placed):
                    continue

                new_seq = seq + (op_id,)
                new_remaining = frozenset(x for x in remaining if x != op_id)

                h_score = partial_hamiltonian_energy(new_seq)

                c_score = completion_makespan(new_seq, new_remaining)

                score = h_score + COMPLETION_SCORE_WEIGHT * (c_score - incumbent_C)

                score += QI_NOISE * rng.random()

                new_beam.append((score, new_seq, new_remaining))

        if not new_beam:
            return None

        new_beam.sort(key=lambda item: item[0])
        beam = new_beam[:QI_BOND_DIM]

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
# 10. BENCHMARK ONE INSTANCE
# ============================================================

def benchmark_instance(instance_name):
    print(f"\nRunning {instance_name}...")

    jobs_data, optimum = load_jobs_data(instance_name)

    rows = []

    # Greedy
    greedy = best_greedy(jobs_data)

    feasible, C_greedy = check_schedule(jobs_data, greedy["starts"])

    rows.append({
        "Instance": instance_name,
        "Method": "Greedy",
        "Makespan": C_greedy,
        "Gap_%": gap_percent(C_greedy, optimum),
        "Runtime_s": greedy["runtime"],
    })

    # QUBO
    qubo_starts, C_qubo, qubo_runtime, qubo_sweeps = compact_qubo_baseline(
        jobs_data,
        greedy["starts"],
        optimum,
        seed=QI_SEED,
    )

    feasible, C_qubo = check_schedule(jobs_data, qubo_starts)

    rows.append({
        "Instance": instance_name,
        "Method": "QUBO",
        "Makespan": C_qubo,
        "Gap_%": gap_percent(C_qubo, optimum),
        "Runtime_s": greedy["runtime"] + qubo_runtime,
    })

    # CP-SAT
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

    # QI Compact-QUBO TN-LNS
    # Warm-start from QUBO, not Local Search.
    qi_starts, C_qi, qi_runtime, qi_sweeps = qi_compact_qubo_tn_lns(
        jobs_data,
        qubo_starts,
        optimum,
        seed=QI_SEED,
    )

    feasible, C_qi = check_schedule(jobs_data, qi_starts)

    rows.append({
        "Instance": instance_name,
        "Method": "QI Compact-QUBO TN-LNS",
        "Makespan": C_qi,
        "Gap_%": gap_percent(C_qi, optimum),
        "Runtime_s": greedy["runtime"] + qubo_runtime + qi_runtime,
    })

    return pd.DataFrame(rows)


# ============================================================
# 11. RUN FULL BENCHMARK
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
# 12. METRIC TABLES
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
# 13. ONLY THREE GRAPHS
# ============================================================

METHOD_ORDER = [
    "Greedy",
    "QUBO",
    "CP-SAT",
    "QI Compact-QUBO TN-LNS",
]

INSTANCE_ORDER = [
    instance for instance in INSTANCE_NAMES
    if instance in results["Instance"].unique()
]


def plot_metric(metric, title, ylabel, logy=False):
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
    ylabel="Runtime seconds",
    logy=True,
)


# ============================================================
# 14. SAVE RESULTS
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
