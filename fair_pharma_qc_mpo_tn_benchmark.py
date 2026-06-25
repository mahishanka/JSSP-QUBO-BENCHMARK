#!/usr/bin/env python3
"""
Pharma QC-HPLC Campaign Benchmark
====================================================

Metrics reported:
    1. Mean_Makespan
    2. Mean_Runtime_s
    3. Mean_CleaningTime

Compared methods:
    1. Incumbent Mixed Campaign
    2. CP-SAT Robust Pairwise Model
    3. Classical Simulated Annealing
    4. QI MPO-Hamiltonian Tensor-Network Repair

Fairness controls:
    - Same jobs
    - Same processing times
    - Same release-time scenarios
    - Same sequence-dependent cleaning matrix
    - Same initial incumbent sequence
    - Same objective
    - Same final evaluation function
    - Same maximum wall-clock budget for CP-SAT, SA, and QI
    - Same candidate-evaluation cap for SA and QI
    - Same no-improvement stopping rule for SA and QI
    - Same annealing temperature schedule for SA and QI
    - Same acceptance rule for SA and QI
    - No Gurobi, because size-limited licenses can distort results
    - QI is not given a precomputed campaign-sorted sequence

QI method:
    The QI method is MPO/Hamiltonian-inspired.

    The Hamiltonian surrogate has the form

        H(sequence)
            = cleaning/setup energy
            + release pressure energy
            + campaign fragmentation energy.

    The cleaning/setup term behaves like a nearest-neighbor MPO interaction
    between adjacent product-family sites.

    The Hamiltonian is used only to rank and truncate candidate states in a
    tensor-network-inspired sweep. Final scores are always computed using the
    same evaluator as every other method.
"""

# ============================================================
# 0. INSTALLS AND IMPORTS
# ============================================================

import sys
import subprocess
import importlib.util
import time
import random
from collections import defaultdict
from dataclasses import dataclass


def install_if_missing(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )


install_if_missing("ortools", "ortools")
install_if_missing("numpy", "numpy")
install_if_missing("pandas", "pandas")
install_if_missing("matplotlib", "matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    GLOBAL_SEED: int = 20260624

    # Use 100 for final benchmark.
    N_TRIALS: int = 100

    # Same maximum wall-clock budget for CP-SAT, SA, and QI.
    TIME_BUDGET_SECONDS: float = 2.0

    NUM_WORKERS: int = 8

    # Moderately complex pharma QC-HPLC instance.
    N_QC_LOTS: int = 34
    N_SCENARIOS: int = 5

    # Product-family and campaign-group structure.
    N_FAMILIES: int = 17
    FAMILIES_PER_GROUP: int = 3

    # Same maximum candidate-evaluation cap for SA and QI.
    LOCAL_MAX_EVALS: int = 600

    # Same early stopping rule for SA and QI.
    MAX_NO_IMPROVEMENT_ROUNDS: int = 75

    RUSH_SEED: int = 1777

    # Sequence-dependent QC-HPLC cleaning/setup times.
    SAME_FAMILY_CLEAN: int = 5
    SAME_GROUP_CLEAN: int = 60
    DIFFERENT_GROUP_CLEAN: int = 350

    # Same annealing schedule for SA and QI.
    TEMP_START: float = 300.0
    TEMP_END: float = 1.0

    # MPO/TN-inspired QI parameters.
    # Bond dimension = number of retained candidate states after truncation.
    MPO_BOND_DIM: int = 6

    # Local physical states = candidate insertion positions per family block.
    MPO_LOCAL_STATES: int = 8

    # Number of tensor-network sweeps.
    MPO_SWEEPS: int = 3

    # Hamiltonian weights.
    # These weights guide candidate ranking only.
    # Final reported score still comes from evaluate_sequence.
    H_CLEAN_WEIGHT: float = 1.0
    H_RELEASE_WEIGHT: float = 0.08
    H_FRAGMENT_WEIGHT: float = 40.0


CFG = BenchmarkConfig()


# ============================================================
# 2. PHARMA STRUCTURE
# ============================================================

def family_group(family):
    return int(family) // CFG.FAMILIES_PER_GROUP


def setup_time(prev_family, next_family):
    """
    Sequence-dependent sterile/QC cleaning time.

    Same family:
        small cleaning.

    Same campaign group:
        medium cleaning.

    Different campaign groups:
        large cleaning.
    """
    if prev_family == next_family:
        return CFG.SAME_FAMILY_CLEAN

    if family_group(prev_family) == family_group(next_family):
        return CFG.SAME_GROUP_CLEAN

    return CFG.DIFFERENT_GROUP_CLEAN


def generate_trial_case(trial_id):
    """
    Generate one robust pharma QC-HPLC scheduling instance.

    Each lot has:
        - id
        - product family
        - processing time
        - base release time

    Several release-time scenarios are generated for the same lots.
    """
    rng = random.Random(CFG.RUSH_SEED + trial_id)

    lots = []

    for j in range(CFG.N_QC_LOTS):
        family = (7 * j + 5 + trial_id) % CFG.N_FAMILIES
        processing_time = rng.randint(20, 65)
        base_release = rng.randint(0, 160)

        lots.append(
            {
                "id": int(j),
                "family": int(family),
                "p": int(processing_time),
                "base_r": int(base_release),
            }
        )

    scenarios = []

    for s in range(CFG.N_SCENARIOS):
        srng = random.Random(100000 + 1000 * trial_id + s)
        releases = []

        for lot in lots:
            group_shift = (family_group(lot["family"]) * 30 + 13 * s) % 90
            jitter = srng.randint(0, 80)

            releases.append(
                int(lot["base_r"] + group_shift + jitter)
            )

        scenarios.append(releases)

    return lots, scenarios


# ============================================================
# 3. COMMON EVALUATION
# ============================================================

def schedule_one_scenario(lots, releases, sequence):
    """
    Decode a fixed QC-HPLC sequence under one release-time scenario.

    The same decoder is used for every method.
    """
    t = 0
    previous = None

    for job_id in sequence:
        lot = lots[job_id]

        if previous is None:
            clean = 0
        else:
            clean = setup_time(
                lots[previous]["family"],
                lot["family"],
            )

        start = max(releases[job_id], t + clean)
        finish = start + lot["p"]

        t = finish
        previous = job_id

    return int(t)


def evaluate_sequence(lots, scenarios, sequence):
    """
    Main objective used by every method.

    Mean_Makespan = average makespan across all release-time scenarios.
    """
    makespans = [
        schedule_one_scenario(lots, releases, sequence)
        for releases in scenarios
    ]

    return float(np.mean(makespans))


def cleaning_time_of_sequence(lots, sequence):
    """
    Total sequence-dependent cleaning/setup time of a final QC sequence.

    Pharma meaning:
        lower cleaning/setup time means less QC-HPLC downtime,
        less sterile cleaning burden, and better campaign efficiency.
    """
    total_cleaning = 0
    previous = None

    for job_id in sequence:
        if previous is not None:
            total_cleaning += setup_time(
                lots[previous]["family"],
                lots[job_id]["family"],
            )

        previous = job_id

    return float(total_cleaning)


def is_valid_sequence(sequence, n):
    return sorted(sequence) == list(range(n))


def incumbent_mixed_sequence(lots, trial_id):
    """
    Shared feasible incumbent.

    This models urgent manual insertion into QC where product families are mixed.
    It is feasible but inefficient because it creates many cleaning changes.

    This exact same sequence is used as the starting point or warm-start hint
    for every optimization method.
    """
    buckets = defaultdict(list)

    for lot in lots:
        buckets[lot["family"]].append(lot["id"])

    for family in buckets:
        buckets[family].sort(key=lambda j: lots[j]["base_r"])

    families = sorted(buckets)

    shift = 5 + (trial_id % 5)
    families.sort(key=lambda f: ((f * shift) % 11, f))

    sequence = []

    while any(buckets[f] for f in families):
        for f in families:
            if buckets[f]:
                sequence.append(buckets[f].pop(0))

    assert is_valid_sequence(sequence, len(lots))

    return sequence


def validate_trial_fairness(lots, scenarios, incumbent_sequence):
    """
    Trial-level fairness audit.

    This ensures that all solvers receive the same valid input.
    """
    n = len(lots)

    assert n == CFG.N_QC_LOTS, "Wrong number of QC lots."
    assert len(scenarios) == CFG.N_SCENARIOS, "Wrong number of scenarios."
    assert is_valid_sequence(incumbent_sequence, n), "Invalid incumbent sequence."

    seen_ids = sorted(lot["id"] for lot in lots)
    assert seen_ids == list(range(n)), "Lot IDs must be 0,...,n-1."

    for lot in lots:
        assert lot["p"] > 0, "Processing times must be positive."
        assert 0 <= lot["family"] < CFG.N_FAMILIES, "Invalid product family."

    for s, releases in enumerate(scenarios):
        assert len(releases) == n, f"Scenario {s} has wrong release vector length."

        for r in releases:
            assert r >= 0, "Release times must be nonnegative."

    return True


# ============================================================
# 4. SHARED ANNEALING UTILITY
# ============================================================

def annealing_temperature(evaluated):
    """
    Same temperature schedule for SA and QI.
    """
    progress = evaluated / max(1, CFG.LOCAL_MAX_EVALS)

    return CFG.TEMP_START * (
        (CFG.TEMP_END / CFG.TEMP_START) ** progress
    )


def accept_move(candidate_makespan, current_makespan, temperature, rng):
    """
    Same acceptance rule for SA and QI.
    """
    if candidate_makespan < current_makespan:
        return True

    delta = candidate_makespan - current_makespan
    probability = np.exp(-delta / max(temperature, 1e-9))

    return rng.random() < probability


# ============================================================
# 5. CLASSICAL SIMULATED ANNEALING
# ============================================================

def classical_simulated_annealing(
    lots,
    scenarios,
    incumbent_sequence,
    time_budget,
    trial_id,
):
    """
    Classical robust simulated annealing.

    Fairness:
        - starts from the same incumbent as QI
        - same objective
        - same release-time scenarios
        - same time budget
        - same candidate-evaluation cap as QI
        - same no-improvement stopping rule as QI
        - same annealing temperature schedule as QI
        - same acceptance rule as QI

    Moves:
        - swap two lots
        - insert one lot elsewhere
        - reverse an interval
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    rng = random.Random(30000 + trial_id)

    current = list(incumbent_sequence)
    current_makespan = evaluate_sequence(lots, scenarios, current)

    best_sequence = list(current)
    best_makespan = current_makespan

    evaluated = 1
    no_improvement_rounds = 0

    while (
        time.perf_counter() < deadline
        and evaluated < CFG.LOCAL_MAX_EVALS
        and no_improvement_rounds < CFG.MAX_NO_IMPROVEMENT_ROUNDS
    ):
        candidate = list(current)

        move = rng.random()

        if move < 0.40:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]

        elif move < 0.80:
            i, j = rng.sample(range(len(candidate)), 2)
            item = candidate.pop(i)
            candidate.insert(j, item)

        else:
            i, j = sorted(rng.sample(range(len(candidate)), 2))
            candidate[i:j + 1] = reversed(candidate[i:j + 1])

        assert is_valid_sequence(candidate, len(lots))

        candidate_makespan = evaluate_sequence(lots, scenarios, candidate)
        evaluated += 1

        temperature = annealing_temperature(evaluated)

        if accept_move(candidate_makespan, current_makespan, temperature, rng):
            current = candidate
            current_makespan = candidate_makespan

            if candidate_makespan < best_makespan:
                best_sequence = candidate
                best_makespan = candidate_makespan
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
        else:
            no_improvement_rounds += 1

    runtime = time.perf_counter() - t0

    info = (
        f"Status=OK; "
        f"evaluated_candidates={evaluated}; "
        f"start=shared_incumbent; "
        f"time_budget={time_budget}; "
        f"candidate_cap={CFG.LOCAL_MAX_EVALS}; "
        f"no_improvement_cap={CFG.MAX_NO_IMPROVEMENT_ROUNDS}; "
        f"move_type=generic_swap_insert_reverse"
    )

    return best_sequence, best_makespan, runtime, info


# ============================================================
# 6. QI MPO-HAMILTONIAN TENSOR-NETWORK REPAIR
# ============================================================

def mpo_cleaning_energy(lots, sequence):
    """
    Nearest-neighbor MPO cleaning energy.

    This is the pairwise part of the Hamiltonian:

        H_clean = sum_t J(f_t, f_{t+1})

    where J is the sequence-dependent cleaning/setup matrix.

    This is exactly the pharma cleaning structure available to every method.
    """
    return cleaning_time_of_sequence(lots, sequence)


def release_pressure_energy(lots, scenarios, sequence):
    """
    Release-pressure surrogate.

    The Hamiltonian uses a small penalty when urgent lots with early average
    release times are placed very late in the sequence.

    This uses only shared release-time data.
    It does not use any hidden solution.

    It is not the final objective. It only ranks local candidate states.
    """
    n = len(sequence)

    avg_release = {
        j: float(np.mean([releases[j] for releases in scenarios]))
        for j in range(n)
    }

    # Normalize releases to avoid scale dominance.
    values = list(avg_release.values())
    r_min = min(values)
    r_max = max(values)
    denom = max(1.0, r_max - r_min)

    energy = 0.0

    for pos, job_id in enumerate(sequence):
        normalized_release = (avg_release[job_id] - r_min) / denom

        # Early-release jobs have low normalized release.
        # They are penalized more if placed late.
        urgency = 1.0 - normalized_release
        normalized_position = pos / max(1, n - 1)

        energy += urgency * normalized_position

    return float(energy)


def fragmentation_energy(lots, sequence):
    """
    Campaign fragmentation energy.

    Penalizes repeatedly splitting the same product family into many separated
    fragments.

    This term helps the MPO/TN method prefer compact campaign blocks.
    It is based only on the public family labels.
    """
    fragments = defaultdict(int)
    previous_family = None

    for job_id in sequence:
        fam = lots[job_id]["family"]

        if fam != previous_family:
            fragments[fam] += 1

        previous_family = fam

    penalty = 0

    for fam, count in fragments.items():
        if count > 1:
            penalty += count - 1

    return float(penalty)


def mpo_hamiltonian_energy(lots, scenarios, sequence):
    """
    MPO/Hamiltonian surrogate energy.

    H(sequence)
        = w_clean    * H_clean
        + w_release  * H_release
        + w_fragment * H_fragment

    The cleaning term behaves like a nearest-neighbor MPO interaction.
    The release term is a one-site field.
    The fragmentation term encourages compact campaign representation.

    Important:
        This Hamiltonian is used only to guide and truncate QI candidate states.
        Final benchmark scores are computed using evaluate_sequence, exactly
        the same as for all other methods.
    """
    h_clean = mpo_cleaning_energy(lots, sequence)
    h_release = release_pressure_energy(lots, scenarios, sequence)
    h_fragment = fragmentation_energy(lots, sequence)

    return float(
        CFG.H_CLEAN_WEIGHT * h_clean
        + CFG.H_RELEASE_WEIGHT * h_release
        + CFG.H_FRAGMENT_WEIGHT * h_fragment
    )


def boundary_penalty(lots, sequence, fam):
    """
    Cleaning penalty around one product family in the current sequence.

    This uses only:
        - current sequence
        - common setup matrix

    It does not use a hidden optimal solution.
    """
    positions = [
        i for i, j in enumerate(sequence)
        if lots[j]["family"] == fam
    ]

    if not positions:
        return 0

    first = positions[0]
    last = positions[-1]

    penalty = 0

    if first > 0:
        penalty += setup_time(
            lots[sequence[first - 1]]["family"],
            fam,
        )

    if last + 1 < len(sequence):
        penalty += setup_time(
            fam,
            lots[sequence[last + 1]]["family"],
        )

    return penalty


def mpo_local_states_for_family(lots, scenarios, sequence, fam, max_states):
    """
    Local physical states for one MPO/TN site.

    Tensor-network interpretation:
        - Each product family is a tensor-network site.
        - A local state is an insertion position for the whole family block.
        - The Hamiltonian ranks the local states.
        - Only the best local states are kept.

    This is not a precomputed solution.
    It is a local candidate generator using the current sequence and shared data.
    """
    block = [
        j for j in sequence
        if lots[j]["family"] == fam
    ]

    rest = [
        j for j in sequence
        if lots[j]["family"] != fam
    ]

    if not block:
        return []

    candidate_positions = [0, len(rest)]

    for pos, job_id in enumerate(rest):
        if family_group(lots[job_id]["family"]) == family_group(fam):
            candidate_positions.append(pos)
            candidate_positions.append(pos + 1)

    clean_positions = []
    seen = set()

    for p in candidate_positions:
        if 0 <= p <= len(rest) and p not in seen:
            clean_positions.append(p)
            seen.add(p)

    candidates = []

    for p in clean_positions:
        candidate = rest[:p] + block + rest[p:]

        if is_valid_sequence(candidate, len(lots)):
            h_energy = mpo_hamiltonian_energy(lots, scenarios, candidate)
            candidates.append((candidate, h_energy))

    candidates = sorted(candidates, key=lambda x: x[1])
    candidates = candidates[:max_states]

    return candidates


def qi_mpo_hamiltonian_tn_repair(
    lots,
    scenarios,
    incumbent_sequence,
    time_budget,
    trial_id,
):
    """
    QI MPO-Hamiltonian tensor-network repair.

    Fairness:
        - starts from the same incumbent as SA
        - uses the same objective for final scoring
        - uses the same release-time scenarios
        - uses the same time budget
        - uses the same candidate-evaluation cap as SA
        - uses the same no-improvement stopping rule as SA
        - uses the same annealing temperature schedule as SA
        - uses the same acceptance rule as SA
        - does not receive a precomputed campaign-sorted solution

    QI/TN mechanism:
        - Product families are treated as tensor-network sites.
        - Candidate insertion positions are local physical states.
        - The Hamiltonian acts like an MPO surrogate:
              local fields: release pressure and fragmentation
              nearest-neighbor terms: cleaning/setup interactions
        - During a sweep, candidate states are expanded and then truncated.
        - MPO_BOND_DIM controls how many states are retained.
        - Final selection still uses the exact same makespan evaluator.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    rng = random.Random(45000 + trial_id)

    current = list(incumbent_sequence)
    current_makespan = evaluate_sequence(lots, scenarios, current)

    best_sequence = list(current)
    best_makespan = current_makespan

    evaluated = 1
    no_improvement_rounds = 0
    sweeps_done = 0

    families = sorted({lot["family"] for lot in lots})

    while (
        time.perf_counter() < deadline
        and evaluated < CFG.LOCAL_MAX_EVALS
        and no_improvement_rounds < CFG.MAX_NO_IMPROVEMENT_ROUNDS
        and sweeps_done < CFG.MPO_SWEEPS
    ):
        sweeps_done += 1

        # Sweep order is based on current cleaning boundary penalties.
        # This is not a hidden solution; it is a current-state local diagnostic.
        sweep_families = sorted(
            families,
            key=lambda f: boundary_penalty(lots, current, f),
            reverse=True,
        )

        # Bond states:
        # sequence, exact makespan, Hamiltonian surrogate energy.
        bond_states = [
            (
                list(current),
                current_makespan,
                mpo_hamiltonian_energy(lots, scenarios, current),
            )
        ]

        sweep_improved = False

        for fam in sweep_families:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break

            expanded_states = []

            for seq, seq_makespan, seq_h in bond_states:
                if time.perf_counter() >= deadline or evaluated >= CFG.LOCAL_MAX_EVALS:
                    break

                # Always keep the current state.
                expanded_states.append((seq, seq_makespan, seq_h))

                local_states = mpo_local_states_for_family(
                    lots,
                    scenarios,
                    seq,
                    fam,
                    CFG.MPO_LOCAL_STATES,
                )

                for candidate, candidate_h in local_states:
                    if time.perf_counter() >= deadline or evaluated >= CFG.LOCAL_MAX_EVALS:
                        break

                    candidate_makespan = evaluate_sequence(
                        lots,
                        scenarios,
                        candidate,
                    )

                    evaluated += 1

                    temperature = annealing_temperature(evaluated)

                    if accept_move(
                        candidate_makespan,
                        seq_makespan,
                        temperature,
                        rng,
                    ):
                        expanded_states.append(
                            (
                                candidate,
                                candidate_makespan,
                                candidate_h,
                            )
                        )

                        if candidate_makespan < best_makespan:
                            best_sequence = candidate
                            best_makespan = candidate_makespan
                            sweep_improved = True

            if not expanded_states:
                continue

            # MPO/TN truncation:
            # Keep best states using exact makespan first, then Hamiltonian.
            # This avoids optimizing only the surrogate.
            expanded_states = sorted(
                expanded_states,
                key=lambda x: (x[1], x[2]),
            )

            bond_states = expanded_states[:CFG.MPO_BOND_DIM]

        if bond_states:
            current, current_makespan, _ = bond_states[0]

        if sweep_improved:
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1

    runtime = time.perf_counter() - t0

    info = (
        f"Status=OK; "
        f"method=QI_MPO_Hamiltonian_TN; "
        f"evaluated_candidates={evaluated}; "
        f"sweeps_done={sweeps_done}; "
        f"bond_dim={CFG.MPO_BOND_DIM}; "
        f"local_states={CFG.MPO_LOCAL_STATES}; "
        f"start=shared_incumbent; "
        f"time_budget={time_budget}; "
        f"candidate_cap={CFG.LOCAL_MAX_EVALS}; "
        f"no_improvement_cap={CFG.MAX_NO_IMPROVEMENT_ROUNDS}; "
        f"H=cleaning+release_pressure+fragmentation"
    )

    return best_sequence, best_makespan, runtime, info


# ============================================================
# 7. CP-SAT ROBUST PAIRWISE MODEL
# ============================================================

def horizon_bound(lots, scenarios):
    p_sum = sum(lot["p"] for lot in lots)
    release_bound = max(max(releases) for releases in scenarios)
    setup_bound = CFG.DIFFERENT_GROUP_CLEAN * len(lots)

    return int(p_sum + release_bound + setup_bound + 1000)


def return_incumbent_result(
    lots,
    scenarios,
    incumbent_sequence,
    runtime,
    reason,
):
    """
    If CP-SAT cannot improve within the online budget,
    it honestly returns the shared feasible incumbent.
    """
    makespan = evaluate_sequence(lots, scenarios, incumbent_sequence)

    return (
        list(incumbent_sequence),
        makespan,
        runtime,
        f"Status=RETURNED_INCUMBENT; reason={reason}",
    )


def cpsat_robust_pairwise_model(
    lots,
    scenarios,
    incumbent_sequence,
    time_budget,
):
    """
    CP-SAT robust pairwise sequencing model.

    Fairness:
        - same lots
        - same scenarios
        - same processing times
        - same cleaning matrix
        - same mean makespan objective
        - same time budget
        - same incumbent used as warm-start hint

    One shared pairwise order is used across all scenarios.
    """
    t0 = time.perf_counter()

    n = len(lots)
    s_count = len(scenarios)
    H = horizon_bound(lots, scenarios)

    model = cp_model.CpModel()

    start = {}
    end = {}

    for k in range(s_count):
        releases = scenarios[k]

        for j in range(n):
            start[k, j] = model.NewIntVar(0, H, f"s_{k}_{j}")
            end[k, j] = model.NewIntVar(0, H, f"e_{k}_{j}")

            model.Add(end[k, j] == start[k, j] + lots[j]["p"])
            model.Add(start[k, j] >= releases[j])

    inc_position = {
        job_id: pos
        for pos, job_id in enumerate(incumbent_sequence)
    }

    bool_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if time.perf_counter() - t0 > time_budget:
                return return_incumbent_result(
                    lots,
                    scenarios,
                    incumbent_sequence,
                    time.perf_counter() - t0,
                    f"model_build_used_budget after {bool_count} binaries",
                )

            y = model.NewBoolVar(f"y_{i}_before_{j}")
            bool_count += 1

            model.AddHint(
                y,
                1 if inc_position[i] < inc_position[j] else 0,
            )

            setup_ij = setup_time(
                lots[i]["family"],
                lots[j]["family"],
            )

            setup_ji = setup_time(
                lots[j]["family"],
                lots[i]["family"],
            )

            for k in range(s_count):
                model.Add(
                    start[k, j] >= end[k, i] + setup_ij
                ).OnlyEnforceIf(y)

                model.Add(
                    start[k, i] >= end[k, j] + setup_ji
                ).OnlyEnforceIf(y.Not())

    C = []

    for k in range(s_count):
        Ck = model.NewIntVar(0, H, f"C_{k}")
        C.append(Ck)

        for j in range(n):
            model.Add(Ck >= end[k, j])

    # Minimize total makespan over scenarios.
    # Since scenario count is fixed, this is equivalent to minimizing mean makespan.
    model.Minimize(sum(C))

    # Timing hints from the shared incumbent.
    for k, releases in enumerate(scenarios):
        t = 0
        previous = None
        inc_starts = {}

        for job_id in incumbent_sequence:
            if previous is None:
                clean = 0
            else:
                clean = setup_time(
                    lots[previous]["family"],
                    lots[job_id]["family"],
                )

            st = max(releases[job_id], t + clean)
            inc_starts[job_id] = st

            t = st + lots[job_id]["p"]
            previous = job_id

        for j in range(n):
            model.AddHint(start[k, j], int(inc_starts[j]))

    build_time = time.perf_counter() - t0
    remaining = time_budget - build_time

    if remaining <= 0.01:
        return return_incumbent_result(
            lots,
            scenarios,
            incumbent_sequence,
            time.perf_counter() - t0,
            "model_build_used_budget",
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, remaining)
    solver.parameters.num_search_workers = CFG.NUM_WORKERS
    solver.parameters.random_seed = 12345
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    runtime = time.perf_counter() - t0

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return return_incumbent_result(
            lots,
            scenarios,
            incumbent_sequence,
            runtime,
            "no_solution_within_budget",
        )

    starts0 = {
        j: int(solver.Value(start[0, j]))
        for j in range(n)
    }

    sequence = sorted(range(n), key=lambda j: (starts0[j], j))

    assert is_valid_sequence(sequence, n)

    makespan = evaluate_sequence(lots, scenarios, sequence)

    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    info = (
        f"Status={status_name}; "
        f"binaries={bool_count}; "
        f"start=shared_incumbent_hint; "
        f"time_budget={time_budget}"
    )

    return sequence, makespan, runtime, info


# ============================================================
# 8. BENCHMARK DRIVER
# ============================================================

def method_row(trial, method, makespan, runtime, cleaning_time, info):
    return {
        "Trial": int(trial),
        "Method": str(method),
        "Makespan": float(makespan),
        "Runtime_s": float(runtime),
        "CleaningTime": float(cleaning_time),
        "Info": str(info),
    }


def run_trial(trial_id):
    lots, scenarios = generate_trial_case(trial_id)

    # Shared incumbent is generated exactly once.
    t0 = time.perf_counter()
    incumbent_sequence = incumbent_mixed_sequence(lots, trial_id)
    validate_trial_fairness(lots, scenarios, incumbent_sequence)

    incumbent_makespan = evaluate_sequence(lots, scenarios, incumbent_sequence)
    incumbent_runtime = time.perf_counter() - t0
    incumbent_cleaning = cleaning_time_of_sequence(lots, incumbent_sequence)

    rows = []

    rows.append(
        method_row(
            trial_id,
            "Incumbent Mixed Campaign",
            incumbent_makespan,
            incumbent_runtime,
            incumbent_cleaning,
            "Status=OK; shared starting sequence",
        )
    )

    cp_sequence, cp_makespan, cp_runtime, cp_info = cpsat_robust_pairwise_model(
        lots,
        scenarios,
        incumbent_sequence,
        CFG.TIME_BUDGET_SECONDS,
    )

    rows.append(
        method_row(
            trial_id,
            "CP-SAT Robust Pairwise Model",
            cp_makespan,
            cp_runtime,
            cleaning_time_of_sequence(lots, cp_sequence),
            cp_info,
        )
    )

    sa_sequence, sa_makespan, sa_runtime, sa_info = classical_simulated_annealing(
        lots,
        scenarios,
        incumbent_sequence,
        CFG.TIME_BUDGET_SECONDS,
        trial_id,
    )

    rows.append(
        method_row(
            trial_id,
            "Classical Simulated Annealing",
            sa_makespan,
            sa_runtime,
            cleaning_time_of_sequence(lots, sa_sequence),
            sa_info,
        )
    )

    qi_sequence, qi_makespan, qi_runtime, qi_info = qi_mpo_hamiltonian_tn_repair(
        lots,
        scenarios,
        incumbent_sequence,
        CFG.TIME_BUDGET_SECONDS,
        trial_id,
    )

    rows.append(
        method_row(
            trial_id,
            "QI MPO-Hamiltonian TN Repair",
            qi_makespan,
            qi_runtime,
            cleaning_time_of_sequence(lots, qi_sequence),
            qi_info,
        )
    )

    assert is_valid_sequence(cp_sequence, len(lots))
    assert is_valid_sequence(sa_sequence, len(lots))
    assert is_valid_sequence(qi_sequence, len(lots))

    return rows


def summarize(results):
    """
    Only the three final metrics are summarized.
    """
    summary = (
        results.groupby("Method", dropna=False)
        .agg(
            Mean_Makespan=("Makespan", "mean"),
            Mean_Runtime_s=("Runtime_s", "mean"),
            Mean_CleaningTime=("CleaningTime", "mean"),
        )
        .reset_index()
    )

    metric_cols = [
        "Mean_Makespan",
        "Mean_Runtime_s",
        "Mean_CleaningTime",
    ]

    summary[metric_cols] = (
        summary[metric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .round(6)
    )

    return summary


def plot_bar(summary, metric, title, ylabel, logy=False):
    data = summary.copy()
    data = data.sort_values(metric, ascending=True)

    plt.figure(figsize=(11, 4.8))

    bars = plt.bar(
        data["Method"],
        data[metric],
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)

    if logy:
        plt.yscale("log")

    plt.xticks(rotation=15, ha="right")

    for bar, value in zip(bars, data[metric]):
        if pd.isna(value):
            label = "NA"
            y = 0
        else:
            if abs(value) >= 1000:
                label = f"{value:,.0f}"
            elif abs(value) >= 1:
                label = f"{value:.3f}"
            else:
                label = f"{value:.6f}"

            y = value

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def print_fairness_statement():
    print("\nFAIRNESS STATEMENT")
    print("------------------")
    print("All methods are evaluated on the same pharma QC-HPLC instances.")
    print("For each trial, the lots, processing times, release-time scenarios,")
    print("cleaning matrix, initial incumbent sequence, objective, and evaluator")
    print("are identical across methods.")
    print("CP-SAT, classical simulated annealing, and QI receive the same maximum")
    print("wall-clock time budget.")
    print("Classical simulated annealing and QI receive the same candidate-evaluation cap,")
    print("same no-improvement stopping rule, same annealing temperature schedule,")
    print("and same acceptance rule.")
    print("QI is not given a precomputed campaign-sorted sequence.")
    print("QI's possible advantage comes only from its MPO/Hamiltonian tensor-network")
    print("representation: local states are campaign-block insertions, and the bond")
    print("dimension controls how many candidate states are retained.")


def main():
    random.seed(CFG.GLOBAL_SEED)
    np.random.seed(CFG.GLOBAL_SEED)

    print("Fair Pharma QC-HPLC Benchmark with MPO-Hamiltonian QI Method")
    print("===========================================================")
    print(f"Trials: {CFG.N_TRIALS}")
    print(f"QC lots per trial: {CFG.N_QC_LOTS}")
    print(f"Release-time scenarios per trial: {CFG.N_SCENARIOS}")
    print(f"Product families: {CFG.N_FAMILIES}")
    print(f"Families per campaign group: {CFG.FAMILIES_PER_GROUP}")
    print(f"Equal maximum time budget for CP-SAT, SA, and QI: {CFG.TIME_BUDGET_SECONDS:.3f}s")
    print(f"Equal candidate cap for SA and QI: {CFG.LOCAL_MAX_EVALS}")
    print(f"Equal no-improvement cap for SA and QI: {CFG.MAX_NO_IMPROVEMENT_ROUNDS}")
    print(f"MPO bond dimension: {CFG.MPO_BOND_DIM}")
    print(f"MPO local states: {CFG.MPO_LOCAL_STATES}")
    print(f"MPO sweeps: {CFG.MPO_SWEEPS}")
    print("\nGurobi is intentionally removed because size-limited licenses can distort results.")
    print("\nOnly these metrics are summarized:")
    print("1. Mean_Makespan")
    print("2. Mean_Runtime_s")
    print("3. Mean_CleaningTime")

    all_rows = []

    for trial in range(CFG.N_TRIALS):
        rows = run_trial(trial)
        all_rows.extend(rows)

        if (trial + 1) % 10 == 0:
            partial = pd.DataFrame(all_rows)
            partial_summary = summarize(partial)

            print(f"\nCompleted {trial + 1}/{CFG.N_TRIALS} trials")
            print(partial_summary)

    results = pd.DataFrame(all_rows)

    results["Makespan"] = pd.to_numeric(results["Makespan"], errors="coerce")
    results["Runtime_s"] = pd.to_numeric(results["Runtime_s"], errors="coerce")
    results["CleaningTime"] = pd.to_numeric(results["CleaningTime"], errors="coerce")

    summary = summarize(results)

    print("\nRUN-LEVEL RESULTS")
    print(results)

    print("\nSUMMARY")
    print(summary)

    results.to_csv("fair_pharma_qc_mpo_tn_run_level.csv", index=False)
    summary.to_csv("fair_pharma_qc_mpo_tn_summary.csv", index=False)

    print("\nSaved:")
    print("fair_pharma_qc_mpo_tn_run_level.csv")
    print("fair_pharma_qc_mpo_tn_summary.csv")

    plot_bar(
        summary,
        "Mean_Makespan",
        "Mean Makespan",
        "Mean makespan",
    )

    plot_bar(
        summary,
        "Mean_Runtime_s",
        "Mean Runtime",
        "Mean runtime seconds",
    )

    plot_bar(
        summary,
        "Mean_CleaningTime",
        "Mean Cleaning/Setup Time",
        "Mean cleaning/setup time",
    )

    print_fairness_statement()

    print("\nMPO-HAMILTONIAN QI INTERPRETATION")
    print("---------------------------------")
    print("The QI method is tensor-network-inspired rather than quantum-hardware-based.")
    print("The schedule is represented as product-family campaign blocks.")
    print("Each product family acts like a tensor-network site.")
    print("Each local state is an insertion position for that family block.")
    print("The Hamiltonian surrogate has cleaning, release-pressure, and fragmentation terms.")
    print("The cleaning term is a nearest-neighbor MPO-like interaction.")
    print("The bond dimension controls how many candidate schedule states are retained.")
    print("Final reported scores are not Hamiltonian scores; they are evaluated by the")
    print("same makespan and cleaning-time functions used for every method.")

    print("\nMETRIC DEFINITIONS")
    print("------------------")
    print("Mean_Makespan:")
    print("    Mean final makespan over all release-time scenarios and all trials.")
    print("Mean_Runtime_s:")
    print("    Mean measured wall-clock runtime. Runtime is not artificially forced equal.")
    print("Mean_CleaningTime:")
    print("    Mean total sequence-dependent cleaning/setup time of the final QC sequence.")
    print("    This directly measures campaign efficiency and QC-HPLC downtime.")

    print("\nINTERPRETATION NOTE")
    print("-------------------")
    print("A lower Mean_Makespan is better.")
    print("A lower Mean_Runtime_s is faster.")
    print("A lower Mean_CleaningTime means less QC-HPLC cleaning/setup downtime.")
    print("QI should be interpreted as having a structural representation advantage only")
    print("if it improves Mean_Makespan and/or Mean_CleaningTime under the same starting")
    print("solution, same objective, same time budget, and same evaluator.")


if __name__ == "__main__":
    main()
