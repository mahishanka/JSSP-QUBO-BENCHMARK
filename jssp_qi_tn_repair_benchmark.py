#!/usr/bin/env python3
"""
Robust Multi-Scenario Rolling-Horizon JSSP Benchmark
====================================================

Scenario:
    Structured production-cell JSSP
    + partial execution frozen
    + stochastic bottleneck machine breakdown
    + stochastic rush-job arrivals
    + stochastic processing-time delays
    + stochastic worker/machine unavailability

Methods:
    1. Greedy Repair
    2. Compact-QUBO Annealing
    3. CP-SAT Repair
    4. Compact-QUBO MPO-TN Repair

Charts:
    1. Robust makespan score
    2. Total runtime over all disruption scenarios
    3. Size metric

The goal is not to claim that CP-SAT fails on ordinary JSSP.
The goal is to test repeated stochastic disruption repair, where CP-SAT must
solve many related full models while Compact-QUBO MPO-TN repairs local affected blocks.
"""

# ============================================================
# 0. COLAB-SAFE INSTALLATION
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


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)


# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================

# Start with only TF-small if you want a quick test:
# INSTANCE_CONFIGS = INSTANCE_CONFIGS[:1]

INSTANCE_CONFIGS = [
    {
        "name": "TF-small",
        "n_cells": 3,
        "jobs_per_cell": 5,
        "machines_per_cell": 3,
        "local_ops_per_job": 5,
        "global_bottleneck_ops": 2,
        "seed": 101,
    },
    {
        "name": "TF-medium",
        "n_cells": 4,
        "jobs_per_cell": 6,
        "machines_per_cell": 4,
        "local_ops_per_job": 6,
        "global_bottleneck_ops": 2,
        "seed": 102,
    },
    {
        "name": "TF-large",
        "n_cells": 5,
        "jobs_per_cell": 7,
        "machines_per_cell": 4,
        "local_ops_per_job": 7,
        "global_bottleneck_ops": 2,
        "seed": 103,
    },
    {
        "name": "TF-xlarge",
        "n_cells": 6,
        "jobs_per_cell": 8,
        "machines_per_cell": 4,
        "local_ops_per_job": 8,
        "global_bottleneck_ops": 2,
        "seed": 104,
    },
]

# Number of stochastic disruption scenarios per instance.
# Increase to 30 or 50 for stronger CP-SAT stress.
NUM_DISRUPTION_SCENARIOS = 20

# CP-SAT is intentionally given a short online repair time.
# Increase/decrease this to control CP-SAT difficulty.
CP_SAT_REPAIR_TIME_LIMIT = 0.03

# Optional longer CP-SAT reference is not used in the plots.
# It is only stored for inspection.
CP_SAT_REFERENCE_TIME_LIMIT = 0.50

# Robust score:
# lower is better.
ROBUST_AVG_WEIGHT = 0.60
ROBUST_WORST_WEIGHT = 0.40

# If a method fails on a scenario, penalize it using greedy makespan.
UNSOLVED_PENALTY_FACTOR = 1.35

# Machine breakdown randomness.
BREAKDOWN_START_FRACTION_RANGE = (0.25, 0.45)
BREAKDOWN_DURATION_FRACTION_RANGE = (0.12, 0.30)

# Rush job randomness.
RUSH_JOBS_PER_CELL_RANGE = (2, 5)
RUSH_LOCAL_OPS_PER_JOB_RANGE = (3, 6)
RUSH_BOTTLENECK_OPS = 2

# Processing-time uncertainty.
DELAY_PROBABILITY = 0.30
DELAY_MULTIPLIER_RANGE = (1.10, 1.60)

# Extra worker/machine unavailability.
EXTRA_UNAVAILABILITY_PROBABILITY = 0.60
EXTRA_UNAVAILABILITY_DURATION_FRACTION_RANGE = (0.05, 0.15)

# Compact-QUBO Annealing settings.
COMPACT_QUBO_BLOCK_SIZE = 8
COMPACT_QUBO_MAX_BLOCKS = 6
COMPACT_QUBO_SWEEPS = 3
COMPACT_QUBO_SA_STEPS = 250
COMPACT_QUBO_START_TEMP = 8.0
COMPACT_QUBO_END_TEMP = 0.05
COMPACT_QUBO_ENERGY_WEIGHT = 1.0
COMPACT_QUBO_MAKESPAN_WEIGHT = 1.0

# Compact-QUBO MPO-TN settings.
TN_BOND_DIM_LIST = [16, 32, 64, 96, 128]
TN_BLOCK_SIZE = 16
TN_MAX_BLOCKS = 12
TN_MAX_SWEEPS = 5
TN_TIME_BUDGET = 1.25
TN_NOISE = 0.10
TN_QAOA_WARM_SAMPLES = 12

# Compact-QUBO Hamiltonian weights.
BROKEN_MACHINE_PRIORITY = 70.0
AFFECTED_JOB_PRIORITY = 40.0
RUSH_JOB_PRIORITY = 100.0
LOW_SLACK_PRIORITY = 0.08
LONG_PROCESSING_PRIORITY = 0.12
LOCAL_PRECEDENCE_PENALTY = 10000.0


# ============================================================
# 3. STRUCTURED INSTANCE GENERATOR
# ============================================================

def generate_tensor_friendly_jssp(
    n_cells,
    jobs_per_cell,
    machines_per_cell,
    local_ops_per_job,
    global_bottleneck_ops=2,
    seed=123,
):
    """
    Generate a structured production-cell JSSP.

    Jobs mostly remain inside one cell but visit shared bottleneck machines.
    """
    rng = random.Random(seed)
    jobs_data = []

    n_local_machines = n_cells * machines_per_cell
    bottleneck_1 = n_local_machines
    bottleneck_2 = n_local_machines + 1

    for cell in range(n_cells):
        cell_machines = list(
            range(cell * machines_per_cell, (cell + 1) * machines_per_cell)
        )

        for _ in range(jobs_per_cell):
            route = []

            for k in range(local_ops_per_job):
                route.append(cell_machines[k % machines_per_cell])

            insert_pos_1 = max(1, local_ops_per_job // 3)
            insert_pos_2 = max(2, 2 * local_ops_per_job // 3)

            route.insert(insert_pos_1, bottleneck_1)

            if global_bottleneck_ops >= 2:
                route.insert(insert_pos_2, bottleneck_2)

            job = []

            for machine in route:
                if machine in [bottleneck_1, bottleneck_2]:
                    p = rng.randint(25, 60)
                else:
                    p = rng.randint(5, 25)

                job.append((machine, p))

            jobs_data.append(job)

    return jobs_data


def describe_instance(jobs_data):
    n_jobs = len(jobs_data)
    n_ops = sum(len(job) for job in jobs_data)
    machines = sorted({m for job in jobs_data for m, _ in job})
    n_machines = len(machines)
    horizon = sum(p for job in jobs_data for _, p in job)

    machine_loads = defaultdict(int)

    for job in jobs_data:
        for machine, p in job:
            machine_loads[machine] += p

    bottleneck_machine = max(machine_loads, key=machine_loads.get)

    return {
        "n_jobs": n_jobs,
        "n_ops": n_ops,
        "n_machines": n_machines,
        "horizon": horizon,
        "bottleneck_machine": bottleneck_machine,
        "machine_loads": dict(machine_loads),
    }


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
        ids_for_job = []

        for index_in_job, (machine, p) in enumerate(job):
            op = {
                "id": op_id,
                "job": job_id,
                "index": index_in_job,
                "machine": machine,
                "p": int(p),
                "name": f"O{job_id}_{index_in_job}",
            }

            ops.append(op)
            ids_for_job.append(op_id)
            machine_ops[machine].append(op_id)

            op_id += 1

        job_ops.append(ids_for_job)

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
            s = starts[a]
            e = s + ops[a]["p"]
            intervals.append((s, e, a))

        intervals.sort()

        for (s1, e1, _), (s2, e2, _) in zip(intervals[:-1], intervals[1:]):
            if e1 > s2:
                return False, makespan

    return True, makespan


def machine_orders_from_starts(jobs_data, starts):
    _, _, machine_ops, _ = build_operations(jobs_data)

    machine_orders = {}

    for machine, ids in machine_ops.items():
        machine_orders[machine] = sorted(ids, key=lambda a: (starts[a], a))

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
    topo = []

    while queue:
        a = queue.popleft()
        topo.append(a)

        for b, _ in adjacency[a]:
            indegree[b] -= 1

            if indegree[b] == 0:
                queue.append(b)

    if len(topo) < n:
        return None, None, False

    starts = {a: 0 for a in range(n)}

    for a in topo:
        for b, weight in adjacency[a]:
            starts[b] = max(starts[b], starts[a] + weight)

    makespan = max(starts[a] + ops[a]["p"] for a in starts)

    return starts, makespan, True


# ============================================================
# 5. GREEDY SCHEDULING
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

        _, chosen_op, start_time = min(candidates, key=lambda x: x[0])

        op = ops[chosen_op]

        starts[chosen_op] = start_time
        job_ready[op["job"]] = start_time + op["p"]
        machine_ready[op["machine"]] = start_time + op["p"]
        next_op[op["job"]] += 1

    feasible, makespan = check_schedule(jobs_data, starts)

    if not feasible:
        raise RuntimeError("Greedy produced infeasible schedule.")

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
# 6. STOCHASTIC DISRUPTION SCENARIO GENERATION
# ============================================================

def completed_operations_before_event(jobs_data, starts, event_time):
    ops, _, _, _ = build_operations(jobs_data)

    frozen_starts = {}

    for op in ops:
        op_id = op["id"]
        finish = starts[op_id] + op["p"]

        if finish <= event_time:
            frozen_starts[op_id] = starts[op_id]

    return frozen_starts


def generate_rush_jobs(config, seed):
    rng = random.Random(seed)

    n_cells = config["n_cells"]
    machines_per_cell = config["machines_per_cell"]

    n_local_machines = n_cells * machines_per_cell
    bottleneck_1 = n_local_machines
    bottleneck_2 = n_local_machines + 1

    rush_jobs = []

    for cell in range(n_cells):
        cell_machines = list(
            range(cell * machines_per_cell, (cell + 1) * machines_per_cell)
        )

        n_rush = rng.randint(*RUSH_JOBS_PER_CELL_RANGE)

        for _ in range(n_rush):
            local_ops = rng.randint(*RUSH_LOCAL_OPS_PER_JOB_RANGE)

            route = []

            for k in range(local_ops):
                route.append(cell_machines[k % machines_per_cell])

            route.insert(1, bottleneck_1)

            if RUSH_BOTTLENECK_OPS >= 2:
                route.append(bottleneck_2)

            job = []

            for machine in route:
                if machine in [bottleneck_1, bottleneck_2]:
                    p = rng.randint(30, 75)
                else:
                    p = rng.randint(8, 30)

                job.append((machine, p))

            rush_jobs.append(job)

    return rush_jobs


def build_initial_repair_orders_with_rush_jobs(
    base_jobs_data,
    repair_jobs_data,
    original_orders,
):
    repair_ops, _, repair_machine_ops, _ = build_operations(repair_jobs_data)

    old_n_ops = sum(len(job) for job in base_jobs_data)

    repair_orders = {
        machine: list(seq)
        for machine, seq in original_orders.items()
    }

    for machine in repair_machine_ops:
        if machine not in repair_orders:
            repair_orders[machine] = []

    for op in repair_ops:
        op_id = op["id"]

        if op_id >= old_n_ops:
            repair_orders[op["machine"]].append(op_id)

    return repair_orders


def make_stochastic_repair_scenario(
    config,
    base_jobs_data,
    original_starts,
    original_orders,
    scenario_seed,
):
    rng = random.Random(scenario_seed)

    base_ops, _, base_machine_ops, _ = build_operations(base_jobs_data)

    original_makespan = max(
        original_starts[a] + base_ops[a]["p"]
        for a in original_starts
    )

    machine_loads = {
        machine: sum(base_ops[a]["p"] for a in ids)
        for machine, ids in base_machine_ops.items()
    }

    broken_machine = max(machine_loads, key=machine_loads.get)

    start_fraction = rng.uniform(*BREAKDOWN_START_FRACTION_RANGE)
    duration_fraction = rng.uniform(*BREAKDOWN_DURATION_FRACTION_RANGE)

    breakdown_start = int(start_fraction * original_makespan)
    breakdown_duration = max(1, int(duration_fraction * original_makespan))
    breakdown_end = breakdown_start + breakdown_duration

    frozen_starts = completed_operations_before_event(
        base_jobs_data,
        original_starts,
        breakdown_start,
    )

    frozen_set = set(frozen_starts.keys())

    # Determine affected original jobs.
    affected_original_jobs = set()

    for op in base_ops:
        if op["id"] in frozen_set:
            continue

        if op["machine"] == broken_machine:
            affected_original_jobs.add(op["job"])

    # Modify processing times for pending affected operations.
    repair_jobs_data = []

    for job_id, job in enumerate(base_jobs_data):
        new_job = []

        for index, (machine, p) in enumerate(job):
            op_id = sum(len(base_jobs_data[j]) for j in range(job_id)) + index

            new_p = int(p)

            if op_id not in frozen_set:
                if (
                    job_id in affected_original_jobs
                    or machine == broken_machine
                    or rng.random() < DELAY_PROBABILITY
                ):
                    multiplier = rng.uniform(*DELAY_MULTIPLIER_RANGE)
                    new_p = max(1, int(round(p * multiplier)))

            new_job.append((machine, new_p))

        repair_jobs_data.append(new_job)

    # Add rush jobs.
    rush_jobs = generate_rush_jobs(config, seed=scenario_seed + 9999)

    original_n_jobs = len(repair_jobs_data)

    for job in rush_jobs:
        repair_jobs_data.append(job)

    rush_job_ids = set(range(original_n_jobs, len(repair_jobs_data)))

    # Optional extra worker/machine unavailability.
    downtimes = [(broken_machine, breakdown_start, breakdown_end)]

    if rng.random() < EXTRA_UNAVAILABILITY_PROBABILITY:
        machines = sorted({m for job in repair_jobs_data for m, _ in job})
        candidate_machines = [m for m in machines if m != broken_machine]

        if candidate_machines:
            extra_machine = rng.choice(candidate_machines)
            extra_start = breakdown_start + rng.randint(0, max(1, breakdown_duration))
            extra_duration = max(
                1,
                int(rng.uniform(*EXTRA_UNAVAILABILITY_DURATION_FRACTION_RANGE) * original_makespan),
            )
            extra_end = extra_start + extra_duration
            downtimes.append((extra_machine, extra_start, extra_end))

    event = {
        "scenario_seed": scenario_seed,
        "event_time": breakdown_start,
        "broken_machine": broken_machine,
        "breakdown_start": breakdown_start,
        "breakdown_end": breakdown_end,
        "breakdown_duration": breakdown_duration,
        "downtimes": downtimes,
        "rush_job_ids": rush_job_ids,
        "affected_original_jobs": affected_original_jobs,
    }

    repair_orders = build_initial_repair_orders_with_rush_jobs(
        base_jobs_data=base_jobs_data,
        repair_jobs_data=repair_jobs_data,
        original_orders=original_orders,
    )

    return repair_jobs_data, repair_orders, frozen_starts, event


# ============================================================
# 7. DYNAMIC REPAIR DECODER
# ============================================================

def shift_past_downtimes(machine, start, duration, event):
    changed = True

    while changed:
        changed = False

        for downtime_machine, down_start, down_end in event["downtimes"]:
            if machine != downtime_machine:
                continue

            end = start + duration

            if start < down_end and end > down_start:
                start = down_end
                changed = True

    return start


def dynamic_schedule_from_machine_orders(
    jobs_data,
    machine_orders,
    frozen_starts,
    event,
):
    ops, job_ops, _, _ = build_operations(jobs_data)

    n = len(ops)

    starts = dict(frozen_starts)
    frozen_set = set(frozen_starts.keys())

    job_position = {}

    for job_id, ids in enumerate(job_ops):
        for pos, op_id in enumerate(ids):
            job_position[op_id] = pos

    machine_position = {}

    for machine, seq in machine_orders.items():
        for pos, op_id in enumerate(seq):
            machine_position[op_id] = pos

    pending = set(range(n)) - frozen_set

    while pending:
        progress = False

        for op_id in list(pending):
            op = ops[op_id]
            job_id = op["job"]
            machine = op["machine"]

            # Job predecessor readiness.
            jpos = job_position[op_id]

            if jpos > 0:
                job_pred = job_ops[job_id][jpos - 1]

                if job_pred not in starts:
                    continue

                job_ready = starts[job_pred] + ops[job_pred]["p"]
            else:
                job_ready = 0

            # Machine predecessor readiness.
            seq = machine_orders[machine]
            mpos = machine_position[op_id]

            if mpos > 0:
                machine_pred = seq[mpos - 1]

                if machine_pred not in starts:
                    continue

                machine_ready = starts[machine_pred] + ops[machine_pred]["p"]
            else:
                machine_ready = 0

            start = max(
                event["event_time"],
                job_ready,
                machine_ready,
            )

            start = shift_past_downtimes(
                machine=machine,
                start=start,
                duration=op["p"],
                event=event,
            )

            starts[op_id] = start
            pending.remove(op_id)
            progress = True

        if not progress:
            return None, None, False

    makespan = max(starts[a] + ops[a]["p"] for a in starts)

    return starts, makespan, True


def check_dynamic_schedule(jobs_data, starts, frozen_starts, event):
    feasible, makespan = check_schedule(jobs_data, starts)

    if not feasible:
        return False, makespan

    ops, _, _, _ = build_operations(jobs_data)

    for op_id, frozen_start in frozen_starts.items():
        if starts[op_id] != frozen_start:
            return False, makespan

    for op in ops:
        op_id = op["id"]

        if op_id not in frozen_starts and starts[op_id] < event["event_time"]:
            return False, makespan

    for op in ops:
        op_id = op["id"]
        s = starts[op_id]
        e = s + op["p"]

        for downtime_machine, down_start, down_end in event["downtimes"]:
            if op["machine"] != downtime_machine:
                continue

            if s < down_end and e > down_start:
                return False, makespan

    return True, makespan


# ============================================================
# 8. METRIC HELPERS
# ============================================================

def compact_qubo_variable_count(jobs_data):
    _, _, machine_ops, _ = build_operations(jobs_data)

    Nq = 0

    for _, ids in machine_ops.items():
        k = len(ids)
        Nq += k * (k - 1) // 2

    return int(Nq)


def estimate_compact_qubo_matrix_metrics(jobs_data):
    _, _, machine_ops, _ = build_operations(jobs_data)

    Nq = compact_qubo_variable_count(jobs_data)
    Q_size = int(Nq * Nq)

    nonzeros = Nq

    for _, ids in machine_ops.items():
        k = len(ids)

        nonzeros += k * (k - 1) // 2

        if k >= 3:
            nonzeros += 3 * (k * (k - 1) * (k - 2) // 6)

    nonzeros = min(nonzeros, Q_size)

    density = nonzeros / Q_size if Q_size > 0 else np.nan

    return Nq, Q_size, density


def estimate_tn_memory_mb(num_ops, bond_dim):
    local_dim = 2
    bytes_used = num_ops * local_dim * bond_dim * bond_dim * 8
    return bytes_used / (1024 ** 2)


# ============================================================
# 9. CP-SAT REPAIR
# ============================================================

def cpsat_repair(
    jobs_data,
    frozen_starts,
    event,
    time_limit,
    hint_starts=None,
):
    start_time = time.perf_counter()

    ops, job_ops, machine_ops, _ = build_operations(jobs_data)

    horizon = sum(op["p"] for op in ops)
    horizon += max(end for _, _, end in event["downtimes"])
    horizon = int(max(1, horizon))

    model = cp_model.CpModel()

    start_vars = {}
    end_vars = {}
    intervals_by_machine = defaultdict(list)

    for op in ops:
        op_id = op["id"]

        s = model.NewIntVar(0, horizon, f"s_{op_id}")
        e = model.NewIntVar(0, horizon, f"e_{op_id}")
        interval = model.NewIntervalVar(s, op["p"], e, f"I_{op_id}")

        start_vars[op_id] = s
        end_vars[op_id] = e
        intervals_by_machine[op["machine"]].append(interval)

        if op_id in frozen_starts:
            model.Add(s == int(frozen_starts[op_id]))
        else:
            model.Add(s >= int(event["event_time"]))

    # Fixed downtime intervals inserted into NoOverlap.
    for downtime_index, (machine, down_start, down_end) in enumerate(event["downtimes"]):
        duration = int(down_end - down_start)

        if duration > 0:
            downtime_interval = model.NewIntervalVar(
                int(down_start),
                int(duration),
                int(down_end),
                f"DOWN_{machine}_{downtime_index}",
            )

            intervals_by_machine[machine].append(downtime_interval)

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
            if op_id in hint_starts:
                model.AddHint(start_vars[op_id], int(hint_starts[op_id]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 123

    status = solver.Solve(model)

    runtime = time.perf_counter() - start_time

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None, None, False, runtime, "NO_SOLUTION"

    starts = {
        op_id: solver.Value(start_vars[op_id])
        for op_id in start_vars
    }

    feasible, C = check_dynamic_schedule(
        jobs_data,
        starts,
        frozen_starts,
        event,
    )

    if not feasible:
        return None, None, False, runtime, "INFEASIBLE_OUTPUT"

    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    return starts, C, True, runtime, status_name


# ============================================================
# 10. BLOCK SELECTION
# ============================================================

def affected_jobs_for_repair(jobs_data, frozen_starts, event):
    ops, _, _, _ = build_operations(jobs_data)

    frozen_set = set(frozen_starts.keys())
    affected_jobs = set(event.get("rush_job_ids", set()))

    downtime_machines = {m for m, _, _ in event["downtimes"]}

    for op in ops:
        op_id = op["id"]

        if op_id in frozen_set:
            continue

        if op["machine"] in downtime_machines:
            affected_jobs.add(op["job"])

    return affected_jobs


def block_priority_score(jobs_data, current_starts, block, frozen_starts, event):
    ops, _, _, _ = build_operations(jobs_data)

    affected_jobs = affected_jobs_for_repair(jobs_data, frozen_starts, event)
    rush_jobs = event.get("rush_job_ids", set())
    downtime_machines = {m for m, _, _ in event["downtimes"]}

    current_C = max(current_starts[a] + ops[a]["p"] for a in current_starts)

    finish = {
        a: current_starts[a] + ops[a]["p"]
        for a in current_starts
    }

    slack = {
        a: max(0, current_C - finish[a])
        for a in current_starts
    }

    score = 0.0

    for op_id in block:
        op = ops[op_id]

        if op["machine"] in downtime_machines:
            score += BROKEN_MACHINE_PRIORITY

        if op["job"] in affected_jobs:
            score += AFFECTED_JOB_PRIORITY

        if op["job"] in rush_jobs:
            score += RUSH_JOB_PRIORITY

        score += LOW_SLACK_PRIORITY * (current_C - slack[op_id])
        score += LONG_PROCESSING_PRIORITY * op["p"]

    return score


def candidate_blocks_for_repair(
    jobs_data,
    current_starts,
    frozen_starts,
    event,
    block_size,
    max_blocks,
    rng,
):
    ops, _, _, _ = build_operations(jobs_data)

    machine_orders = machine_orders_from_starts(jobs_data, current_starts)

    frozen_set = set(frozen_starts.keys())
    affected_jobs = affected_jobs_for_repair(jobs_data, frozen_starts, event)
    downtime_machines = {m for m, _, _ in event["downtimes"]}

    candidates = []

    def add_block(machine, block, bonus):
        block = [op_id for op_id in block if op_id not in frozen_set]

        if len(block) < 4:
            return

        score = bonus
        score += block_priority_score(jobs_data, current_starts, block, frozen_starts, event)
        score += rng.random()

        candidates.append({
            "machine": machine,
            "block": block,
            "score": score,
        })

    step = max(1, block_size // 3)

    # Downtime-machine blocks.
    for machine in downtime_machines:
        if machine not in machine_orders:
            continue

        seq = [
            op_id for op_id in machine_orders[machine]
            if op_id not in frozen_set
        ]

        for lo in range(0, len(seq), step):
            hi = min(len(seq), lo + block_size)
            add_block(machine, seq[lo:hi], bonus=180.0)

    # Affected-job blocks on all machines.
    for machine, seq in machine_orders.items():
        positions = [
            i for i, op_id in enumerate(seq)
            if op_id not in frozen_set and ops[op_id]["job"] in affected_jobs
        ]

        for pos in positions:
            half = block_size // 2
            lo = max(0, pos - half)
            hi = min(len(seq), lo + block_size)
            lo = max(0, hi - block_size)
            add_block(machine, seq[lo:hi], bonus=100.0)

    # General pending blocks.
    for machine, seq in machine_orders.items():
        pending_seq = [op_id for op_id in seq if op_id not in frozen_set]

        for lo in range(0, len(pending_seq), step):
            hi = min(len(pending_seq), lo + block_size)
            add_block(machine, pending_seq[lo:hi], bonus=10.0)

    seen = set()
    unique = []

    for c in candidates:
        key = (c["machine"], tuple(c["block"]))

        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique.sort(key=lambda c: c["score"], reverse=True)

    return unique[:max_blocks]


def replace_selected_block_order(full_sequence, block, permutation):
    block_set = set(block)
    permutation = list(permutation)

    new_sequence = []
    inserted = False

    for op_id in full_sequence:
        if op_id in block_set:
            if not inserted:
                new_sequence.extend(permutation)
                inserted = True
            continue

        new_sequence.append(op_id)

    return new_sequence


def evaluate_block_permutation(
    jobs_data,
    current_orders,
    machine,
    block,
    permutation,
    frozen_starts,
    event,
):
    new_orders = {
        m: list(seq)
        for m, seq in current_orders.items()
    }

    new_orders[machine] = replace_selected_block_order(
        current_orders[machine],
        block,
        permutation,
    )

    starts, C, feasible = dynamic_schedule_from_machine_orders(
        jobs_data,
        new_orders,
        frozen_starts,
        event,
    )

    return starts, C, feasible, new_orders


# ============================================================
# 11. COMPACT-QUBO HAMILTONIAN
# ============================================================

def compact_qubo_block_terms(
    jobs_data,
    current_starts,
    block,
    frozen_starts,
    event,
):
    ops, job_ops, _, _ = build_operations(jobs_data)

    block = list(block)
    block_set = set(block)

    affected_jobs = affected_jobs_for_repair(jobs_data, frozen_starts, event)
    rush_jobs = event.get("rush_job_ids", set())
    downtime_machines = {m for m, _, _ in event["downtimes"]}

    current_C = max(
        current_starts[a] + ops[a]["p"]
        for a in current_starts
    )

    finish = {
        a: current_starts[a] + ops[a]["p"]
        for a in current_starts
    }

    slack = {
        a: max(0, current_C - finish[a])
        for a in current_starts
    }

    must_precede = set()

    for ids in job_ops:
        local_ids = [op_id for op_id in ids if op_id in block_set]

        for i, a in enumerate(local_ids):
            for b in local_ids[i + 1:]:
                must_precede.add((a, b))

    pair_weight = {}

    for a in block:
        for b in block:
            if a == b:
                continue

            op_a = ops[a]
            op_b = ops[b]

            weight = 0.0

            # Bad if we reverse job precedence.
            if (b, a) in must_precede:
                weight += LOCAL_PRECEDENCE_PENALTY

            priority_a = 0.0
            priority_b = 0.0

            if op_a["machine"] in downtime_machines:
                priority_a += BROKEN_MACHINE_PRIORITY

            if op_b["machine"] in downtime_machines:
                priority_b += BROKEN_MACHINE_PRIORITY

            if op_a["job"] in affected_jobs:
                priority_a += AFFECTED_JOB_PRIORITY

            if op_b["job"] in affected_jobs:
                priority_b += AFFECTED_JOB_PRIORITY

            if op_a["job"] in rush_jobs:
                priority_a += RUSH_JOB_PRIORITY

            if op_b["job"] in rush_jobs:
                priority_b += RUSH_JOB_PRIORITY

            priority_a += LOW_SLACK_PRIORITY * (current_C - slack[a])
            priority_b += LOW_SLACK_PRIORITY * (current_C - slack[b])

            priority_a += LONG_PROCESSING_PRIORITY * op_a["p"]
            priority_b += LONG_PROCESSING_PRIORITY * op_b["p"]

            # Encourage high-priority operation earlier.
            weight -= 0.50 * (priority_a - priority_b)

            pair_weight[(a, b)] = weight

    return pair_weight


def compact_qubo_sequence_energy(sequence, pair_weight):
    sequence = list(sequence)

    energy = 0.0

    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            a = sequence[i]
            b = sequence[j]
            energy += pair_weight.get((a, b), 0.0)

    return energy


def local_precedence_feasible_sequence(jobs_data, block, sequence):
    _, job_ops, _, _ = build_operations(jobs_data)

    block_set = set(block)
    pos = {op_id: i for i, op_id in enumerate(sequence)}

    for ids in job_ops:
        local_ids = [op_id for op_id in ids if op_id in block_set]

        for i, a in enumerate(local_ids):
            for b in local_ids[i + 1:]:
                if pos[a] > pos[b]:
                    return False

    return True


# ============================================================
# 12. COMPACT-QUBO ANNEALING
# ============================================================

def compact_qubo_anneal_block(
    jobs_data,
    current_starts,
    frozen_starts,
    event,
    machine,
    block,
    rng,
):
    current_orders = machine_orders_from_starts(jobs_data, current_starts)

    feasible, current_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    if not feasible:
        return None

    pair_weight = compact_qubo_block_terms(
        jobs_data,
        current_starts,
        block,
        frozen_starts,
        event,
    )

    current_sequence = tuple(block)
    current_energy = compact_qubo_sequence_energy(current_sequence, pair_weight)

    current_objective = (
        COMPACT_QUBO_ENERGY_WEIGHT * current_energy
        + COMPACT_QUBO_MAKESPAN_WEIGHT * current_C
    )

    best = {
        "sequence": current_sequence,
        "makespan": current_C,
        "starts": current_starts,
        "objective": current_objective,
    }

    k = len(block)

    for step in range(COMPACT_QUBO_SA_STEPS):
        alpha = step / max(1, COMPACT_QUBO_SA_STEPS - 1)

        temperature = COMPACT_QUBO_START_TEMP * (
            COMPACT_QUBO_END_TEMP / COMPACT_QUBO_START_TEMP
        ) ** alpha

        candidate = list(current_sequence)

        move_type = rng.random()

        if move_type < 0.45:
            i, j = rng.sample(range(k), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]

        elif move_type < 0.80:
            i, j = rng.sample(range(k), 2)
            item = candidate.pop(i)
            candidate.insert(j, item)

        else:
            i, j = sorted(rng.sample(range(k), 2))
            candidate[i:j + 1] = reversed(candidate[i:j + 1])

        candidate = tuple(candidate)

        if not local_precedence_feasible_sequence(jobs_data, block, candidate):
            continue

        candidate_energy = compact_qubo_sequence_energy(candidate, pair_weight)

        cand_starts, cand_C, cand_feasible, _ = evaluate_block_permutation(
            jobs_data,
            current_orders,
            machine,
            block,
            candidate,
            frozen_starts,
            event,
        )

        if not cand_feasible:
            continue

        candidate_objective = (
            COMPACT_QUBO_ENERGY_WEIGHT * candidate_energy
            + COMPACT_QUBO_MAKESPAN_WEIGHT * cand_C
        )

        delta = candidate_objective - current_objective

        if delta <= 0:
            accept = True
        else:
            accept_probability = np.exp(-delta / max(temperature, 1e-9))
            accept = rng.random() < accept_probability

        if accept:
            current_sequence = candidate
            current_energy = candidate_energy
            current_objective = candidate_objective

        if cand_C < best["makespan"]:
            best = {
                "sequence": candidate,
                "makespan": cand_C,
                "starts": cand_starts,
                "objective": candidate_objective,
            }

    if best["makespan"] < current_C:
        return best

    return None


def compact_qubo_annealing_repair(
    jobs_data,
    initial_starts,
    frozen_starts,
    event,
    seed=0,
):
    rng = random.Random(seed)

    start_time = time.perf_counter()

    current_starts = dict(initial_starts)

    feasible, current_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    if not feasible:
        return None, None, False, time.perf_counter() - start_time

    for _ in range(COMPACT_QUBO_SWEEPS):
        blocks = candidate_blocks_for_repair(
            jobs_data,
            current_starts,
            frozen_starts,
            event,
            block_size=COMPACT_QUBO_BLOCK_SIZE,
            max_blocks=COMPACT_QUBO_MAX_BLOCKS,
            rng=rng,
        )

        if not blocks:
            break

        best_move = None

        for block_info in blocks:
            candidate = compact_qubo_anneal_block(
                jobs_data,
                current_starts,
                frozen_starts,
                event,
                machine=block_info["machine"],
                block=block_info["block"],
                rng=rng,
            )

            if candidate is not None:
                if best_move is None or candidate["makespan"] < best_move["makespan"]:
                    best_move = candidate

        if best_move is None:
            break

        current_starts = best_move["starts"]
        current_C = best_move["makespan"]

    runtime = time.perf_counter() - start_time

    feasible, final_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    return current_starts, final_C, feasible, runtime


# ============================================================
# 13. LR-QAOA-INSPIRED WARM STARTS
# ============================================================

def lr_qaoa_inspired_initial_sequences(
    block,
    pair_weight,
    rng,
    num_samples=12,
):
    block = list(block)
    samples = []

    priority = {}

    for a in block:
        score = 0.0

        for b in block:
            if a == b:
                continue

            score -= pair_weight.get((a, b), 0.0)

        priority[a] = score

    base = tuple(sorted(block, key=lambda x: priority[x], reverse=True))
    samples.append(base)

    for _ in range(num_samples - 1):
        seq = list(base)
        temperature = rng.uniform(0.5, 2.0)

        for _ in range(max(2, len(seq) // 2)):
            i, j = rng.sample(range(len(seq)), 2)

            old_seq = tuple(seq)
            old_E = compact_qubo_sequence_energy(old_seq, pair_weight)

            seq[i], seq[j] = seq[j], seq[i]

            new_seq = tuple(seq)
            new_E = compact_qubo_sequence_energy(new_seq, pair_weight)

            delta = new_E - old_E

            if delta > 0:
                accept_prob = np.exp(-delta / max(temperature, 1e-9))

                if rng.random() > accept_prob:
                    seq[i], seq[j] = seq[j], seq[i]

        samples.append(tuple(seq))

    unique = []
    seen = set()

    for sample in samples:
        if sample not in seen:
            seen.add(sample)
            unique.append(sample)

    return unique


# ============================================================
# 14. COMPACT-QUBO MPO/TN REPAIR
# ============================================================

def compact_qubo_mpo_tn_block_solver(
    jobs_data,
    current_starts,
    frozen_starts,
    event,
    machine,
    block,
    bond_dim,
    rng,
):
    ops, job_ops, _, _ = build_operations(jobs_data)

    current_orders = machine_orders_from_starts(jobs_data, current_starts)

    feasible, current_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    if not feasible:
        return None

    block = list(block)
    block_set = set(block)

    pair_weight = compact_qubo_block_terms(
        jobs_data,
        current_starts,
        block,
        frozen_starts,
        event,
    )

    local_predecessors = {op_id: set() for op_id in block}

    for ids in job_ops:
        seen = []

        for op_id in ids:
            if op_id in block_set:
                local_predecessors[op_id].update(
                    x for x in seen if x in block_set
                )
                seen.append(op_id)

    current_position = {
        op_id: i
        for i, op_id in enumerate(current_orders[machine])
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

        cand_starts, cand_C, feasible, _ = evaluate_block_permutation(
            jobs_data,
            current_orders,
            machine,
            block,
            full_sequence,
            frozen_starts,
            event,
        )

        if feasible:
            value = cand_C
        else:
            value = current_C + 10**6

        completion_cache[key] = value

        return value

    warm_sequences = lr_qaoa_inspired_initial_sequences(
        block=block,
        pair_weight=pair_weight,
        rng=rng,
        num_samples=TN_QAOA_WARM_SAMPLES,
    )

    best = None

    # Evaluate warm starts.
    for seq in warm_sequences:
        if not local_precedence_feasible_sequence(jobs_data, block, seq):
            continue

        cand_starts, cand_C, feasible, cand_orders = evaluate_block_permutation(
            jobs_data,
            current_orders,
            machine,
            block,
            seq,
            frozen_starts,
            event,
        )

        if feasible and cand_C < current_C:
            if best is None or cand_C < best["makespan"]:
                best = {
                    "starts": cand_starts,
                    "makespan": cand_C,
                    "orders": cand_orders,
                    "sequence": seq,
                }

    # MPO/MPS-style beam search.
    beam = [(0.0, tuple(), frozenset(block))]

    for _ in range(len(block)):
        new_beam = []

        for _, seq, remaining in beam:
            placed = set(seq)

            for op_id in remaining:
                if not local_predecessors[op_id].issubset(placed):
                    continue

                new_seq = seq + (op_id,)
                new_remaining = frozenset(x for x in remaining if x != op_id)

                qubo_E = compact_qubo_sequence_energy(new_seq, pair_weight)
                comp_C = completion_makespan(new_seq, new_remaining)

                score = (
                    qubo_E
                    + (comp_C - current_C)
                    + TN_NOISE * rng.random()
                )

                new_beam.append((score, new_seq, new_remaining))

        if not new_beam:
            break

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:bond_dim]

    for _, seq, remaining in beam:
        if remaining:
            continue

        cand_starts, cand_C, feasible, cand_orders = evaluate_block_permutation(
            jobs_data,
            current_orders,
            machine,
            block,
            seq,
            frozen_starts,
            event,
        )

        if feasible and cand_C < current_C:
            if best is None or cand_C < best["makespan"]:
                best = {
                    "starts": cand_starts,
                    "makespan": cand_C,
                    "orders": cand_orders,
                    "sequence": seq,
                }

    return best


def compact_qubo_mpo_tn_repair(
    jobs_data,
    initial_starts,
    frozen_starts,
    event,
    seed=0,
):
    rng = random.Random(seed)

    start_time = time.perf_counter()
    deadline = start_time + TN_TIME_BUDGET

    current_starts = dict(initial_starts)

    feasible, current_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    if not feasible:
        return None, None, False, 0.0, np.nan, 0, np.nan

    ops, _, _, _ = build_operations(jobs_data)
    num_ops = len(ops)

    bond_dim_used = TN_BOND_DIM_LIST[0]
    total_sweeps = 0

    for bond_dim in TN_BOND_DIM_LIST:
        bond_dim_used = bond_dim

        improved_at_this_bond = False

        for _ in range(TN_MAX_SWEEPS):
            if time.perf_counter() > deadline:
                break

            blocks = candidate_blocks_for_repair(
                jobs_data,
                current_starts,
                frozen_starts,
                event,
                block_size=TN_BLOCK_SIZE,
                max_blocks=TN_MAX_BLOCKS,
                rng=rng,
            )

            if not blocks:
                break

            best_move = None

            for block_info in blocks:
                if time.perf_counter() > deadline:
                    break

                candidate = compact_qubo_mpo_tn_block_solver(
                    jobs_data,
                    current_starts,
                    frozen_starts,
                    event,
                    machine=block_info["machine"],
                    block=block_info["block"],
                    bond_dim=bond_dim,
                    rng=rng,
                )

                if candidate is not None:
                    if best_move is None or candidate["makespan"] < best_move["makespan"]:
                        best_move = candidate

            total_sweeps += 1

            if best_move is None:
                break

            if best_move["makespan"] < current_C:
                current_starts = best_move["starts"]
                current_C = best_move["makespan"]
                improved_at_this_bond = True
            else:
                break

        if improved_at_this_bond:
            break

    runtime = time.perf_counter() - start_time

    feasible, final_C = check_dynamic_schedule(
        jobs_data,
        current_starts,
        frozen_starts,
        event,
    )

    memory_mb = estimate_tn_memory_mb(num_ops, bond_dim_used)

    return (
        current_starts,
        final_C,
        feasible,
        runtime,
        bond_dim_used,
        total_sweeps,
        memory_mb,
    )


# ============================================================
# 15. BENCHMARK ROWS
# ============================================================

def make_row(
    instance,
    scenario_id,
    method,
    makespan,
    feasible,
    runtime,
    penalty_makespan,
    n_jobs,
    n_machines,
    n_ops,
    rush_jobs,
    size_metric,
    Nq=np.nan,
    Q_size=np.nan,
    Q_density=np.nan,
    bond_dim=np.nan,
    tn_sweeps=np.nan,
    tn_memory_mb=np.nan,
    info="",
):
    robust_makespan = makespan if feasible else penalty_makespan

    return {
        "Instance": instance,
        "Scenario": scenario_id,
        "Method": method,
        "Makespan": makespan if feasible else np.nan,
        "Robust_Makespan": robust_makespan,
        "Feasible": bool(feasible),
        "Runtime_s": runtime,
        "Jobs": n_jobs,
        "Machines": n_machines,
        "Operations": n_ops,
        "Rush_Jobs": rush_jobs,
        "Size_Metric": size_metric,
        "Nq": Nq,
        "Q_Size": Q_size,
        "Q_Density": Q_density,
        "Bond_Dim": bond_dim,
        "TN_Sweeps": tn_sweeps,
        "TN_Memory_MB": tn_memory_mb,
        "Info": info,
    }


# ============================================================
# 16. BENCHMARK ONE INSTANCE
# ============================================================

def benchmark_instance(config):
    instance_name = config["name"]

    base_jobs_data = generate_tensor_friendly_jssp(
        n_cells=config["n_cells"],
        jobs_per_cell=config["jobs_per_cell"],
        machines_per_cell=config["machines_per_cell"],
        local_ops_per_job=config["local_ops_per_job"],
        global_bottleneck_ops=config["global_bottleneck_ops"],
        seed=config["seed"],
    )

    base_initial = best_greedy(base_jobs_data)
    original_starts = base_initial["starts"]
    original_orders = machine_orders_from_starts(base_jobs_data, original_starts)

    rows = []

    print(f"\nInstance {instance_name}")
    print("Base:", describe_instance(base_jobs_data))

    for scenario_id in range(NUM_DISRUPTION_SCENARIOS):
        scenario_seed = config["seed"] * 10000 + scenario_id

        (
            repair_jobs_data,
            repair_orders,
            frozen_starts,
            event,
        ) = make_stochastic_repair_scenario(
            config=config,
            base_jobs_data=base_jobs_data,
            original_starts=original_starts,
            original_orders=original_orders,
            scenario_seed=scenario_seed,
        )

        stats = describe_instance(repair_jobs_data)

        n_jobs = stats["n_jobs"]
        n_machines = stats["n_machines"]
        n_ops = stats["n_ops"]
        rush_jobs = len(event["rush_job_ids"])

        Nq, Q_size, Q_density = estimate_compact_qubo_matrix_metrics(repair_jobs_data)
        cp_var_count = 2 * n_ops + 1

        # ----------------------------------------------------
        # Greedy Repair
        # ----------------------------------------------------
        t0 = time.perf_counter()

        greedy_starts, greedy_C, greedy_feasible = dynamic_schedule_from_machine_orders(
            repair_jobs_data,
            repair_orders,
            frozen_starts,
            event,
        )

        greedy_runtime = time.perf_counter() - t0

        if not greedy_feasible:
            greedy_C = np.nan
            penalty_makespan = 10**9
        else:
            penalty_makespan = UNSOLVED_PENALTY_FACTOR * greedy_C

        rows.append(
            make_row(
                instance=instance_name,
                scenario_id=scenario_id,
                method="Greedy Repair",
                makespan=greedy_C,
                feasible=greedy_feasible,
                runtime=greedy_runtime,
                penalty_makespan=penalty_makespan,
                n_jobs=n_jobs,
                n_machines=n_machines,
                n_ops=n_ops,
                rush_jobs=rush_jobs,
                size_metric=n_ops,
                info=f"rule={base_initial['rule']}",
            )
        )

        if not greedy_feasible:
            continue

        # ----------------------------------------------------
        # Compact-QUBO Annealing
        # ----------------------------------------------------
        qubo_starts, qubo_C, qubo_feasible, qubo_runtime = compact_qubo_annealing_repair(
            repair_jobs_data,
            greedy_starts,
            frozen_starts,
            event,
            seed=scenario_seed + 111,
        )

        rows.append(
            make_row(
                instance=instance_name,
                scenario_id=scenario_id,
                method="Compact-QUBO Annealing",
                makespan=qubo_C,
                feasible=qubo_feasible,
                runtime=qubo_runtime,
                penalty_makespan=penalty_makespan,
                n_jobs=n_jobs,
                n_machines=n_machines,
                n_ops=n_ops,
                rush_jobs=rush_jobs,
                size_metric=Nq,
                Nq=Nq,
                Q_size=Q_size,
                Q_density=Q_density,
                info="compact ordering QUBO + annealing",
            )
        )

        # ----------------------------------------------------
        # CP-SAT Repair
        # ----------------------------------------------------
        cp_starts, cp_C, cp_feasible, cp_runtime, cp_status = cpsat_repair(
            repair_jobs_data,
            frozen_starts,
            event,
            time_limit=CP_SAT_REPAIR_TIME_LIMIT,
            hint_starts=greedy_starts,
        )

        rows.append(
            make_row(
                instance=instance_name,
                scenario_id=scenario_id,
                method="CP-SAT Repair",
                makespan=cp_C,
                feasible=cp_feasible,
                runtime=cp_runtime,
                penalty_makespan=penalty_makespan,
                n_jobs=n_jobs,
                n_machines=n_machines,
                n_ops=n_ops,
                rush_jobs=rush_jobs,
                size_metric=cp_var_count,
                info=f"status={cp_status}; limit={CP_SAT_REPAIR_TIME_LIMIT}s",
            )
        )

        # ----------------------------------------------------
        # Compact-QUBO MPO-TN Repair
        # ----------------------------------------------------
        (
            tn_starts,
            tn_C,
            tn_feasible,
            tn_runtime,
            tn_bond_dim,
            tn_sweeps,
            tn_memory_mb,
        ) = compact_qubo_mpo_tn_repair(
            repair_jobs_data,
            greedy_starts,
            frozen_starts,
            event,
            seed=scenario_seed + 999,
        )

        rows.append(
            make_row(
                instance=instance_name,
                scenario_id=scenario_id,
                method="Compact-QUBO MPO-TN",
                makespan=tn_C,
                feasible=tn_feasible,
                runtime=tn_runtime,
                penalty_makespan=penalty_makespan,
                n_jobs=n_jobs,
                n_machines=n_machines,
                n_ops=n_ops,
                rush_jobs=rush_jobs,
                size_metric=tn_bond_dim,
                bond_dim=tn_bond_dim,
                tn_sweeps=tn_sweeps,
                tn_memory_mb=tn_memory_mb,
                info="compact QUBO Hamiltonian + LR-QAOA warm starts + MPO/TN beam",
            )
        )

        print(
            f"  scenario {scenario_id + 1:02d}/{NUM_DISRUPTION_SCENARIOS}: "
            f"Greedy={greedy_C}, CP={cp_C if cp_feasible else 'FAIL'}, "
            f"TN={tn_C if tn_feasible else 'FAIL'}"
        )

    return rows


# ============================================================
# 17. RUN FULL BENCHMARK
# ============================================================

all_rows = []

for config in INSTANCE_CONFIGS:
    try:
        rows = benchmark_instance(config)
        all_rows.extend(rows)
    except Exception as error:
        print(f"Skipping {config['name']} because of error: {error}")


results = pd.DataFrame(all_rows)

numeric_columns = [
    "Scenario",
    "Makespan",
    "Robust_Makespan",
    "Runtime_s",
    "Jobs",
    "Machines",
    "Operations",
    "Rush_Jobs",
    "Size_Metric",
    "Nq",
    "Q_Size",
    "Q_Density",
    "Bond_Dim",
    "TN_Sweeps",
    "TN_Memory_MB",
]

for col in numeric_columns:
    if col in results.columns:
        results[col] = pd.to_numeric(results[col], errors="coerce")


# ============================================================
# 18B. SCORE TABLE FOR THE THREE MAIN METRICS
# ============================================================

"""
This table reports the three main benchmark metrics:

    1. Robust_Makespan_Score
    2. Total_Runtime_s
    3. Size_Metric

For all three metrics, lower is better.

We also compute ranks:
    rank 1 = best method for that instance and metric.
"""

score_table = summary[
    [
        "Instance",
        "Method",
        "Robust_Makespan_Score",
        "Total_Runtime_s",
        "Size_Metric",
    ]
].copy()

# Rank each method inside each instance.
# Lower value is better for all three metrics.
score_table["Makespan_Rank"] = (
    score_table
    .groupby("Instance")["Robust_Makespan_Score"]
    .rank(method="min", ascending=True)
)

score_table["Runtime_Rank"] = (
    score_table
    .groupby("Instance")["Total_Runtime_s"]
    .rank(method="min", ascending=True)
)

score_table["Size_Rank"] = (
    score_table
    .groupby("Instance")["Size_Metric"]
    .rank(method="min", ascending=True)
)

# Average rank across the three metrics.
score_table["Average_Rank"] = (
    score_table[
        [
            "Makespan_Rank",
            "Runtime_Rank",
            "Size_Rank",
        ]
    ]
    .mean(axis=1)
)

# Optional normalized score.
# For each instance and each metric:
#     best value gets 1.0
#     larger values get proportionally worse scores.
# Lower final score is still better.
def normalized_metric_score(group, column):
    values = group[column].astype(float)
    best = values.min()

    if best <= 0 or np.isnan(best):
        return values / values.max()

    return values / best


score_table["Makespan_Normalized"] = (
    score_table
    .groupby("Instance", group_keys=False)
    .apply(lambda g: normalized_metric_score(g, "Robust_Makespan_Score"))
)

score_table["Runtime_Normalized"] = (
    score_table
    .groupby("Instance", group_keys=False)
    .apply(lambda g: normalized_metric_score(g, "Total_Runtime_s"))
)

score_table["Size_Normalized"] = (
    score_table
    .groupby("Instance", group_keys=False)
    .apply(lambda g: normalized_metric_score(g, "Size_Metric"))
)

score_table["Combined_Normalized_Score"] = (
    score_table[
        [
            "Makespan_Normalized",
            "Runtime_Normalized",
            "Size_Normalized",
        ]
    ]
    .mean(axis=1)
)

# Round columns for clean display.
for col in [
    "Robust_Makespan_Score",
    "Total_Runtime_s",
    "Size_Metric",
    "Makespan_Rank",
    "Runtime_Rank",
    "Size_Rank",
    "Average_Rank",
    "Makespan_Normalized",
    "Runtime_Normalized",
    "Size_Normalized",
    "Combined_Normalized_Score",
]:
    score_table[col] = pd.to_numeric(score_table[col], errors="coerce").round(4)

# Sort by instance, then average rank.
score_table = score_table.sort_values(
    ["Instance", "Average_Rank", "Combined_Normalized_Score"]
).reset_index(drop=True)

print("\nTHREE-METRIC SCORE TABLE")
print(score_table)


# A compact winner table for each instance.
winner_table = (
    score_table
    .sort_values(["Instance", "Average_Rank", "Combined_Normalized_Score"])
    .groupby("Instance", as_index=False)
    .first()
)

winner_table = winner_table[
    [
        "Instance",
        "Method",
        "Robust_Makespan_Score",
        "Total_Runtime_s",
        "Size_Metric",
        "Average_Rank",
        "Combined_Normalized_Score",
    ]
].copy()

print("\nBEST METHOD BY THREE-METRIC SCORE")
print(winner_table)


# Save these score tables.
score_table.to_csv("three_metric_score_table.csv", index=False)
winner_table.to_csv("three_metric_winner_table.csv", index=False)

print("\nSaved score tables:")
print("three_metric_score_table.csv")
print("three_metric_winner_table.csv")


# ============================================================
# 19. ONLY THREE BAR CHARTS
# ============================================================

METHOD_ORDER = [
    "Greedy Repair",
    "Compact-QUBO Annealing",
    "CP-SAT Repair",
    "Compact-QUBO MPO-TN",
]

INSTANCE_ORDER = [config["name"] for config in INSTANCE_CONFIGS]


def plot_bar(metric, title, ylabel, logy=False):
    pivot = summary.pivot_table(
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
        figsize=(12, 5),
        width=0.82,
    )

    ax.set_title(title)
    ax.set_xlabel("Structured stochastic repair instance")
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


# Chart 1: robust makespan score.
plot_bar(
    metric="Robust_Makespan_Score",
    title="Robust Makespan Score over Stochastic Disruption Scenarios",
    ylabel="Robust makespan score",
    logy=False,
)

# Chart 2: total runtime.
plot_bar(
    metric="Total_Runtime_s",
    title="Total Runtime over All Disruption Scenarios",
    ylabel="Total runtime seconds",
    logy=False,
)

# Chart 3: size metric.
plot_bar(
    metric="Size_Metric",
    title="Size Metric: Ops, Compact-QUBO Nq, CP-SAT Vars, TN Bond Dimension",
    ylabel="Size metric",
    logy=True,
)


# ============================================================
# 20. SAVE CSV FILES
# ============================================================

results.to_csv("robust_multiscenario_run_level_results.csv", index=False)
summary.to_csv("robust_multiscenario_summary_metrics.csv", index=False)

print("\nSaved files:")
print("robust_multiscenario_run_level_results.csv")
print("robust_multiscenario_summary_metrics.csv")

print("\nUseful columns to inspect:")
print("Robust_Makespan_Score, Total_Runtime_s, Size_Metric, Feasibility_Rate_%, Solved_Scenarios")
