#!/usr/bin/env python3
"""
!pip -q install ortools dimod dwave-neal pandas matplotlib numpy
JSP-QUBO Benchmark

This script compares two QUBO formulations for the Job-Shop Scheduling Problem:

1. Time-indexed QUBO
2. Compact disjunctive QUBO

It also uses OR-Tools CP-SAT as a classical reference solver.

Outputs:
    outputs/jsp_qubo_benchmark_results.csv
    outputs/jsp_qubo_reduction_ratios.csv
    outputs/qubo_variable_count_comparison.png
    outputs/qubo_quadratic_terms_comparison.png
    outputs/qubo_runtime_comparison.png
"""

import argparse
import itertools
import math
import time
from collections import defaultdict
from pathlib import Path

import dimod
import matplotlib.pyplot as plt
import neal
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


# ============================================================
# 1. Benchmark instance: FT06-style JSP instance
# ============================================================

# Each job is a list of (machine, processing_time).
# This is the standard FT06 instance.
# Known optimum makespan = 55.
JOBS_DATA = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]

KNOWN_OPTIMUM = 55


# ============================================================
# 2. JSP utilities
# ============================================================

def build_operations(jobs_data):
    """
    Convert jobs_data into operation objects and useful index sets.

    Returns
    -------
    ops : list[dict]
        List of all operations.
    job_ops : list[list[int]]
        Operation IDs grouped by job.
    machine_ops : dict[int, list[int]]
        Operation IDs grouped by machine.
    precedences : list[tuple[int, int]]
        Pairs (a, b) meaning operation a must finish before operation b starts.
    """
    ops = []
    job_ops = []
    machine_ops = defaultdict(list)
    op_id = 0

    for j, job in enumerate(jobs_data):
        ids = []

        for k, (machine, processing_time) in enumerate(job):
            op = {
                "id": op_id,
                "job": j,
                "index": k,
                "machine": machine,
                "p": processing_time,
                "name": f"O{j}_{k}",
            }

            ops.append(op)
            ids.append(op_id)
            machine_ops[machine].append(op_id)
            op_id += 1

        job_ops.append(ids)

    precedences = []
    for ids in job_ops:
        for a, b in zip(ids[:-1], ids[1:]):
            precedences.append((a, b))

    return ops, job_ops, machine_ops, precedences


def compute_windows(jobs_data, C):
    """
    Compute earliest and latest start windows using job precedence only.

    For each operation a, we compute:
        L_a = earliest possible start time from previous operations in the job,
        U_a = latest possible start time allowed by makespan C.

    These windows reduce the number of binary variables in the QUBO.
    """
    ops, job_ops, _, _ = build_operations(jobs_data)

    L = {}
    U = {}

    for j, job in enumerate(jobs_data):
        prefix = 0

        for k, (_, processing_time) in enumerate(job):
            op_id = job_ops[j][k]

            L[op_id] = prefix

            remaining_time = sum(job[q][1] for q in range(k, len(job)))
            U[op_id] = C - remaining_time

            prefix += processing_time

    infeasible_ops = [op_id for op_id in L if L[op_id] > U[op_id]]

    return L, U, infeasible_ops


def check_schedule(jobs_data, starts, C=None):
    """
    Check whether a decoded schedule is feasible.

    Parameters
    ----------
    jobs_data : list
        JSP instance.
    starts : dict[int, int]
        Dictionary mapping operation ID to start time.
    C : int or None
        Optional fixed makespan bound.

    Returns
    -------
    feasible : bool
    makespan : int or None
    message : str
    """
    ops, _, machine_ops, precedences = build_operations(jobs_data)

    if starts is None:
        return False, None, "No decoded schedule"

    expected_ids = {op["id"] for op in ops}
    if set(starts.keys()) != expected_ids:
        return False, None, "Missing start times"

    makespan = max(starts[op["id"]] + op["p"] for op in ops)

    if C is not None and makespan > C:
        return False, makespan, f"Makespan {makespan} exceeds C={C}"

    # Check job precedence constraints.
    for a, b in precedences:
        if starts[b] < starts[a] + ops[a]["p"]:
            return (
                False,
                makespan,
                f"Precedence violation: {ops[a]['name']} before {ops[b]['name']}",
            )

    # Check machine non-overlap constraints.
    for machine, ids in machine_ops.items():
        intervals = []

        for op_id in ids:
            op = ops[op_id]
            start = starts[op_id]
            end = start + op["p"]
            intervals.append((start, end, op_id))

        intervals.sort()

        for (s1, e1, a), (s2, e2, b) in zip(intervals[:-1], intervals[1:]):
            if e1 > s2:
                return (
                    False,
                    makespan,
                    f"Machine {machine} overlap: {ops[a]['name']} and {ops[b]['name']}",
                )

    return True, makespan, "Feasible"


# ============================================================
# 3. QUBO builder
# ============================================================

class QUBOBuilder:
    """
    Helper class for building QUBO objectives.

    The QUBO has the form

        E(x) = offset + sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j,

    where x_i are binary variables.
    """

    def __init__(self):
        self.Q = defaultdict(float)
        self.offset = 0.0
        self.variables = set()

    def add_var(self, name):
        self.variables.add(name)
        return name

    def add_qubo_term(self, u, v, coeff):
        """
        Add coeff * u * v to the QUBO.
        """
        if abs(coeff) < 1e-12:
            return

        self.variables.add(u)
        self.variables.add(v)

        key = (u, v) if str(u) <= str(v) else (v, u)
        self.Q[key] += float(coeff)

    def add_square(self, terms, const=0.0, weight=1.0):
        """
        Add a squared penalty:

            weight * (const + sum_i coeff_i x_i)^2.

        Since x_i is binary, x_i^2 = x_i.
        """
        coeffs = defaultdict(float)

        for var, coeff in terms:
            coeffs[var] += coeff
            self.variables.add(var)

        items = [(var, coeff) for var, coeff in coeffs.items() if abs(coeff) > 1e-12]

        self.offset += weight * const * const

        # Linear terms from x_i^2 = x_i and constant interaction.
        for var, a in items:
            self.add_qubo_term(var, var, weight * (a * a + 2 * const * a))

        # Quadratic cross terms.
        for i in range(len(items)):
            var_i, a = items[i]

            for j in range(i + 1, len(items)):
                var_j, b = items[j]
                self.add_qubo_term(var_i, var_j, weight * 2 * a * b)

    def to_bqm(self):
        """
        Convert the QUBO dictionary to a dimod BinaryQuadraticModel.
        """
        return dimod.BinaryQuadraticModel.from_qubo(dict(self.Q), offset=self.offset)

    def stats(self):
        """
        Return size and coefficient statistics for the QUBO.
        """
        bqm = self.to_bqm()

        coeffs = []
        coeffs.extend(abs(v) for v in bqm.linear.values() if abs(v) > 1e-12)
        coeffs.extend(abs(v) for v in bqm.quadratic.values() if abs(v) > 1e-12)

        coeff_ratio = max(coeffs) / min(coeffs) if coeffs else np.nan

        n = bqm.num_variables
        q = len(bqm.quadratic)
        density = 0 if n <= 1 else q / (n * (n - 1) / 2)

        return {
            "binary_vars": n,
            "quadratic_terms": q,
            "density": density,
            "coeff_ratio": coeff_ratio,
            "offset": bqm.offset,
        }


def binary_int_vars(qb, prefix, max_value):
    """
    Encode an integer in [0, max_value] using binary variables.

    Returns a linear expression:
        value = sum coeff_i * bit_i.
    """
    if max_value <= 0:
        return []

    bits = math.ceil(math.log2(max_value + 1))
    terms = []

    for r in range(bits):
        var = qb.add_var(f"{prefix}_b{r}")
        terms.append((var, 2 ** r))

    return terms


def expr_neg(terms):
    return [(var, -coeff) for var, coeff in terms]


def expr_add(*exprs):
    out = []

    for expr in exprs:
        out.extend(expr)

    return out


# ============================================================
# 4. Baseline time-indexed QUBO
# ============================================================

def build_time_indexed_qubo(
    jobs_data,
    C,
    lam_start=1.0,
    lam_prec=1.0,
    lam_mach=1.0,
):
    """
    Build the baseline fixed-makespan time-indexed QUBO.

    Variable:
        x_{a,t} = 1 if operation a starts at time t.

    Energy:
        E = E_start + E_prec + E_mach.

    A feasible schedule should correspond to energy E = 0.
    """
    ops, _, machine_ops, precedences = build_operations(jobs_data)
    L, U, infeasible_ops = compute_windows(jobs_data, C)

    if infeasible_ops:
        raise ValueError(f"C={C} is infeasible by job windows alone.")

    qb = QUBOBuilder()
    x = {}

    # Create start-time variables.
    for op in ops:
        a = op["id"]

        for t in range(L[a], U[a] + 1):
            x[(a, t)] = qb.add_var(f"x_{a}_{t}")

    # Each operation starts exactly once:
    #     (1 - sum_t x_{a,t})^2
    for op in ops:
        a = op["id"]
        terms = [(x[(a, t)], -1) for t in range(L[a], U[a] + 1)]
        qb.add_square(terms, const=1, weight=lam_start)

    # Job precedence:
    # If a before b, forbid u < t + p_a.
    for a, b in precedences:
        p_a = ops[a]["p"]

        for t in range(L[a], U[a] + 1):
            for u in range(L[b], U[b] + 1):
                if u < t + p_a:
                    qb.add_qubo_term(x[(a, t)], x[(b, u)], lam_prec)

    # Machine non-overlap:
    # For operations on the same machine, forbid overlapping intervals.
    for _, ids in machine_ops.items():
        for a, b in itertools.combinations(ids, 2):
            p_a = ops[a]["p"]
            p_b = ops[b]["p"]

            for t in range(L[a], U[a] + 1):
                for u in range(L[b], U[b] + 1):
                    overlap = (t < u + p_b) and (u < t + p_a)

                    if overlap:
                        qb.add_qubo_term(x[(a, t)], x[(b, u)], lam_mach)

    meta = {
        "type": "time_indexed",
        "L": L,
        "U": U,
        "x": x,
    }

    return qb, meta


def decode_time_indexed_sample(jobs_data, sample, meta):
    """
    Decode a time-indexed QUBO sample into start times.
    """
    ops, _, _, _ = build_operations(jobs_data)

    L = meta["L"]
    U = meta["U"]
    x = meta["x"]

    starts = {}

    for op in ops:
        a = op["id"]
        chosen_times = []

        for t in range(L[a], U[a] + 1):
            if sample.get(x[(a, t)], 0) == 1:
                chosen_times.append(t)

        if len(chosen_times) != 1:
            return None

        starts[a] = chosen_times[0]

    return starts


# ============================================================
# 5. Compact disjunctive QUBO
# ============================================================

def build_compact_disjunctive_qubo(
    jobs_data,
    C,
    lam_window=1.0,
    lam_prec=1.0,
    lam_mach=1.0,
):
    """
    Build the compact fixed-makespan disjunctive QUBO.

    Start times are encoded directly:

        s_a = L_a + binary offset.

    For two operations on the same machine, we use one ordering variable:

        y_ab = 1 means a before b,
        y_ab = 0 means b before a.
    """
    ops, _, machine_ops, precedences = build_operations(jobs_data)
    L, U, infeasible_ops = compute_windows(jobs_data, C)

    if infeasible_ops:
        raise ValueError(f"C={C} is infeasible by job windows alone.")

    qb = QUBOBuilder()

    start_expr = {}
    start_bits = {}

    # Start-time encoding and window enforcement.
    for op in ops:
        a = op["id"]
        W = U[a] - L[a]

        offset_terms = binary_int_vars(qb, f"s_{a}", W)
        slack_terms = binary_int_vars(qb, f"win_slack_{a}", W)

        start_expr[a] = (L[a], offset_terms)
        start_bits[a] = offset_terms

        # Enforce:
        #     offset_a + slack_a = W
        # This prevents the offset from exceeding W.
        qb.add_square(
            terms=offset_terms + slack_terms,
            const=-W,
            weight=lam_window,
        )

    # Job precedence:
    #     s_b - s_a - p_a >= 0
    # Convert to equality using slack:
    #     s_b - s_a - p_a - r_ab = 0
    for a, b in precedences:
        p_a = ops[a]["p"]

        const_a, terms_a = start_expr[a]
        const_b, terms_b = start_expr[b]

        r_max = max(0, U[b] - L[a] - p_a)
        r_terms = binary_int_vars(qb, f"prec_slack_{a}_{b}", r_max)

        terms = expr_add(terms_b, expr_neg(terms_a), expr_neg(r_terms))
        const = const_b - const_a - p_a

        qb.add_square(terms, const=const, weight=lam_prec)

    # Machine non-overlap:
    # For each pair on the same machine, either a before b or b before a.
    order_vars = {}

    for _, ids in machine_ops.items():
        for a, b in itertools.combinations(ids, 2):
            p_a = ops[a]["p"]
            p_b = ops[b]["p"]

            const_a, terms_a = start_expr[a]
            const_b, terms_b = start_expr[b]

            y = qb.add_var(f"y_{a}_{b}")
            order_vars[(a, b)] = y

            # Case 1:
            # y = 1 means a before b.
            #
            # Active condition:
            #     s_b - s_a - p_a >= 0
            #
            # Big-M equality:
            #     s_b - s_a - p_a + M_ab(1 - y) - rho_ab = 0
            M_ab = max(0, U[a] + p_a - L[b])
            rho1_max = max(0, U[b] - L[a] - p_a + M_ab)
            rho1_terms = binary_int_vars(qb, f"mach_slack_{a}_{b}_1", rho1_max)

            terms1 = expr_add(
                terms_b,
                expr_neg(terms_a),
                [(y, -M_ab)],
                expr_neg(rho1_terms),
            )
            const1 = const_b - const_a - p_a + M_ab

            qb.add_square(terms1, const=const1, weight=lam_mach)

            # Case 2:
            # y = 0 means b before a.
            #
            # Active condition:
            #     s_a - s_b - p_b >= 0
            #
            # Big-M equality:
            #     s_a - s_b - p_b + M_ba y - rho_ba = 0
            M_ba = max(0, U[b] + p_b - L[a])
            rho2_max = max(0, U[a] - L[b] - p_b + M_ba)
            rho2_terms = binary_int_vars(qb, f"mach_slack_{b}_{a}_2", rho2_max)

            terms2 = expr_add(
                terms_a,
                expr_neg(terms_b),
                [(y, M_ba)],
                expr_neg(rho2_terms),
            )
            const2 = const_a - const_b - p_b

            qb.add_square(terms2, const=const2, weight=lam_mach)

    meta = {
        "type": "compact_disjunctive",
        "L": L,
        "U": U,
        "start_bits": start_bits,
        "order_vars": order_vars,
    }

    return qb, meta


def decode_compact_sample(jobs_data, sample, meta):
    """
    Decode a compact disjunctive QUBO sample into start times.
    """
    ops, _, _, _ = build_operations(jobs_data)

    L = meta["L"]
    start_bits = meta["start_bits"]

    starts = {}

    for op in ops:
        a = op["id"]
        value = L[a]

        for var, coeff in start_bits[a]:
            value += coeff * int(sample.get(var, 0))

        starts[a] = value

    return starts


# ============================================================
# 6. Solvers
# ============================================================

def solve_qubo_with_neal(bqm, num_reads=100, num_sweeps=1000, seed=1):
    """
    Solve a QUBO using D-Wave's neal simulated annealing sampler.
    """
    sampler = neal.SimulatedAnnealingSampler()

    start_time = time.perf_counter()

    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )

    elapsed = time.perf_counter() - start_time

    best_sample = sampleset.first.sample
    best_energy = bqm.energy(best_sample)

    return {
        "sample": best_sample,
        "energy": best_energy,
        "time": elapsed,
        "sampleset": sampleset,
    }


def solve_jsp_cpsat_fixed_C(jobs_data, C, time_limit=5.0):
    """
    CP-SAT reference feasibility check for fixed makespan C.
    """
    model = cp_model.CpModel()

    all_tasks = {}
    machine_to_intervals = defaultdict(list)

    for j, job in enumerate(jobs_data):
        for k, (machine, processing_time) in enumerate(job):
            start = model.NewIntVar(0, C - processing_time, f"start_{j}_{k}")
            end = model.NewIntVar(0, C, f"end_{j}_{k}")
            interval = model.NewIntervalVar(
                start,
                processing_time,
                end,
                f"interval_{j}_{k}",
            )

            all_tasks[(j, k)] = (start, end, interval, machine, processing_time)
            machine_to_intervals[machine].append(interval)

            model.Add(end <= C)

    # Job precedence.
    for j, job in enumerate(jobs_data):
        for k in range(len(job) - 1):
            model.Add(all_tasks[(j, k + 1)][0] >= all_tasks[(j, k)][1])

    # Machine non-overlap.
    for _, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    start_time = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start_time

    feasible = status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    starts = None

    if feasible:
        ops, _, _, _ = build_operations(jobs_data)
        starts = {}

        for op in ops:
            j = op["job"]
            k = op["index"]
            starts[op["id"]] = solver.Value(all_tasks[(j, k)][0])

    return {
        "status": solver.StatusName(status),
        "feasible": feasible,
        "starts": starts,
        "time": elapsed,
    }


def solve_jsp_cpsat_optimize(jobs_data, time_limit=20.0):
    """
    CP-SAT reference optimization: minimize makespan.
    """
    model = cp_model.CpModel()

    horizon = sum(processing_time for job in jobs_data for _, processing_time in job)

    all_tasks = {}
    machine_to_intervals = defaultdict(list)

    makespan = model.NewIntVar(0, horizon, "makespan")

    for j, job in enumerate(jobs_data):
        for k, (machine, processing_time) in enumerate(job):
            start = model.NewIntVar(0, horizon, f"start_{j}_{k}")
            end = model.NewIntVar(0, horizon, f"end_{j}_{k}")
            interval = model.NewIntervalVar(
                start,
                processing_time,
                end,
                f"interval_{j}_{k}",
            )

            all_tasks[(j, k)] = (start, end, interval, machine, processing_time)
            machine_to_intervals[machine].append(interval)

    # Job precedence.
    for j, job in enumerate(jobs_data):
        for k in range(len(job) - 1):
            model.Add(all_tasks[(j, k + 1)][0] >= all_tasks[(j, k)][1])

    # Machine non-overlap.
    for _, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    # Makespan definition.
    for j, job in enumerate(jobs_data):
        last_k = len(job) - 1
        model.Add(makespan >= all_tasks[(j, last_k)][1])

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    start_time = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start_time

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {
            "status": solver.StatusName(status),
            "makespan": None,
            "starts": None,
            "time": elapsed,
        }

    ops, _, _, _ = build_operations(jobs_data)

    starts = {}

    for op in ops:
        j = op["job"]
        k = op["index"]
        starts[op["id"]] = solver.Value(all_tasks[(j, k)][0])

    return {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "starts": starts,
        "time": elapsed,
    }


# ============================================================
# 7. Benchmark for one fixed makespan C
# ============================================================

def benchmark_fixed_C(
    jobs_data,
    C,
    num_reads=100,
    num_sweeps=1000,
    seed=1,
    cpsat_time_limit=5.0,
):
    """
    Run CP-SAT, time-indexed QUBO, and compact disjunctive QUBO
    for one fixed makespan value C.
    """
    rows = []

    # CP-SAT reference.
    cpsat = solve_jsp_cpsat_fixed_C(
        jobs_data,
        C,
        time_limit=cpsat_time_limit,
    )

    if cpsat["starts"] is not None:
        cp_feasible, cp_makespan, cp_msg = check_schedule(
            jobs_data,
            cpsat["starts"],
            C,
        )
    else:
        cp_feasible, cp_makespan, cp_msg = False, None, cpsat["status"]

    rows.append(
        {
            "C": C,
            "model": "CP-SAT fixed C",
            "solver": "OR-Tools CP-SAT",
            "binary_vars": np.nan,
            "quadratic_terms": np.nan,
            "density": np.nan,
            "coeff_ratio": np.nan,
            "energy": np.nan,
            "feasible_found": cp_feasible,
            "makespan": cp_makespan,
            "time_sec": cpsat["time"],
            "status": cp_msg,
        }
    )

    # Baseline time-indexed QUBO.
    try:
        qb_ti, meta_ti = build_time_indexed_qubo(jobs_data, C)
        bqm_ti = qb_ti.to_bqm()
        stats_ti = qb_ti.stats()

        sol_ti = solve_qubo_with_neal(
            bqm_ti,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
        )

        starts_ti = decode_time_indexed_sample(
            jobs_data,
            sol_ti["sample"],
            meta_ti,
        )

        feasible_ti, makespan_ti, msg_ti = check_schedule(
            jobs_data,
            starts_ti,
            C,
        )

        rows.append(
            {
                "C": C,
                "model": "Time-indexed QUBO",
                "solver": "neal simulated annealing",
                **stats_ti,
                "energy": sol_ti["energy"],
                "feasible_found": feasible_ti,
                "makespan": makespan_ti,
                "time_sec": sol_ti["time"],
                "status": msg_ti,
            }
        )

    except Exception as exc:
        rows.append(
            {
                "C": C,
                "model": "Time-indexed QUBO",
                "solver": "neal simulated annealing",
                "binary_vars": np.nan,
                "quadratic_terms": np.nan,
                "density": np.nan,
                "coeff_ratio": np.nan,
                "energy": np.nan,
                "feasible_found": False,
                "makespan": np.nan,
                "time_sec": np.nan,
                "status": f"ERROR: {exc}",
            }
        )

    # Compact disjunctive QUBO.
    try:
        qb_cd, meta_cd = build_compact_disjunctive_qubo(jobs_data, C)
        bqm_cd = qb_cd.to_bqm()
        stats_cd = qb_cd.stats()

        sol_cd = solve_qubo_with_neal(
            bqm_cd,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
        )

        starts_cd = decode_compact_sample(
            jobs_data,
            sol_cd["sample"],
            meta_cd,
        )

        feasible_cd, makespan_cd, msg_cd = check_schedule(
            jobs_data,
            starts_cd,
            C,
        )

        rows.append(
            {
                "C": C,
                "model": "Compact disjunctive QUBO",
                "solver": "neal simulated annealing",
                **stats_cd,
                "energy": sol_cd["energy"],
                "feasible_found": feasible_cd,
                "makespan": makespan_cd,
                "time_sec": sol_cd["time"],
                "status": msg_cd,
            }
        )

    except Exception as exc:
        rows.append(
            {
                "C": C,
                "model": "Compact disjunctive QUBO",
                "solver": "neal simulated annealing",
                "binary_vars": np.nan,
                "quadratic_terms": np.nan,
                "density": np.nan,
                "coeff_ratio": np.nan,
                "energy": np.nan,
                "feasible_found": False,
                "makespan": np.nan,
                "time_sec": np.nan,
                "status": f"ERROR: {exc}",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. Plotting and saving
# ============================================================

def save_comparison_plot(
    qubo_results,
    y_column,
    ylabel,
    title,
    output_path,
    show_plot=False,
):
    """
    Save a comparison plot for the two QUBO models.
    """
    plt.figure(figsize=(8, 5))

    for model_name, group in qubo_results.groupby("model"):
        plt.plot(group["C"], group[y_column], marker="o", label=model_name)

    plt.xlabel("Fixed makespan C")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()

    plt.close()


def compute_reduction_ratios(results, C_values):
    """
    Compute size-reduction ratios comparing time-indexed QUBO
    with compact disjunctive QUBO.
    """
    qubo_results = results[
        results["model"].isin(["Time-indexed QUBO", "Compact disjunctive QUBO"])
    ].copy()

    ratio_rows = []

    for C in C_values:
        sub = qubo_results[qubo_results["C"] == C]

        ti = sub[sub["model"] == "Time-indexed QUBO"]
        cd = sub[sub["model"] == "Compact disjunctive QUBO"]

        if len(ti) == 1 and len(cd) == 1:
            ti = ti.iloc[0]
            cd = cd.iloc[0]

            ratio_rows.append(
                {
                    "C": C,
                    "var_reduction_TI_over_compact": (
                        ti["binary_vars"] / cd["binary_vars"]
                        if cd["binary_vars"]
                        else np.nan
                    ),
                    "quad_reduction_TI_over_compact": (
                        ti["quadratic_terms"] / cd["quadratic_terms"]
                        if cd["quadratic_terms"]
                        else np.nan
                    ),
                    "time_indexed_feasible": ti["feasible_found"],
                    "compact_feasible": cd["feasible_found"],
                    "time_indexed_energy": ti["energy"],
                    "compact_energy": cd["energy"],
                }
            )

    return pd.DataFrame(ratio_rows)


# ============================================================
# 9. Main runner
# ============================================================

def run_benchmark(args):
    """
    Run the full benchmark.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=" * 70)
    print("JSP-QUBO Benchmark")
    print("=" * 70)
    print(f"Known FT06 optimum makespan: {KNOWN_OPTIMUM}")
    print(f"Tested C values: {args.c_values}")
    print(f"Simulated annealing reads: {args.num_reads}")
    print(f"Simulated annealing sweeps: {args.num_sweeps}")
    print(f"Output directory: {output_dir}")
    print()

    print("Running CP-SAT optimization reference...")
    opt_ref = solve_jsp_cps
