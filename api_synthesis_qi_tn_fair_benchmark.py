#!/usr/bin/env python3
"""
Fair API Synthesis Campaign Scheduling Benchmark
================================================

This script refocuses the earlier pharma QC-HPLC benchmark toward the
API Synthesis Campaign Scheduling problem described in the Market-X style
problem analysis.

Pharma scheduling questions represented here:
    1. Which API product/batch should be produced first?
    2. Which reactor/equipment should be assigned to each batch step?
    3. How can equipment idle time and cleaning/changeovers be reduced?
    4. How can due dates and demand forecasts be respected?
    5. How can synthesis, purification, drying, and packaging be coordinated?

Stages:
    SYNTHESIS -> PURIFICATION -> DRYING -> PACKAGING

Metrics compared:
    1. Mean_ThroughputKgPerTime       higher is better
    2. Mean_Makespan                  lower is better
    3. Mean_CleaningChangeoverTime    lower is better
    4. Mean_DueDateHitRate            higher is better

Compared methods:
    1. Incumbent Mixed Campaign
    2. CP-SAT Flexible Equipment Model
    3. Classical Structured Simulated Annealing
    4. QI MPO-Hamiltonian Tensor-Network Campaign Repair

Fairness controls:
    - Same API batches/products
    - Same four-stage route for every batch
    - Same compatible equipment sets
    - Same processing times
    - Same release-time, due-date, and demand scenarios
    - Same sequence-dependent cleaning/changeover matrix
    - Same initial incumbent plan
    - Same final schedule decoder
    - Same final evaluation function and four reported metrics
    - Same maximum wall-clock budget for CP-SAT, SA, and QI
    - Same candidate-evaluation cap for SA and QI
    - Same no-improvement stopping rule for SA and QI
    - Same annealing temperature schedule for SA and QI
    - Same acceptance rule for SA and QI
    - No Gurobi, because size-limited licenses can distort results
    - QI is not given a precomputed campaign-sorted sequence

QI method:
    The QI method is MPO/Hamiltonian-inspired.

    Stronger fair QI version in this file:
        - keeps the same time budget, candidate cap, no-improvement rule,
          annealing schedule, acceptance rule, decoder, and final evaluator.
        - improves only the QI representation: product-block local states, campaign-group contractions,
          critical-batch one-site states, boundary-repair two-site states,
          bottleneck-aware equipment states, and cheaper MPO-style surrogate energies.

    The Hamiltonian surrogate has the form

        H(plan)
            = cleaning/changeover energy
            + due-date pressure energy
            + demand-throughput pressure energy
            + equipment-load balance energy
            + campaign fragmentation energy.

    The cleaning/changeover term is a nearest-neighbor MPO-like interaction
    between adjacent campaign states on shared equipment. The due-date and
    demand terms act like one-site fields. Candidate states are truncated by
    a bond dimension during a tensor-network-inspired sweep.

    Important:
        The Hamiltonian is used only to rank and truncate candidate states.
        Final reported scores are always computed by the same campaign decoder
        and evaluator used for every method.
"""

# ============================================================
# 0. INSTALLS AND IMPORTS
# ============================================================

import sys
import subprocess
import importlib.util
import time
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


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
    GLOBAL_SEED: int = 20260626

    # Use 100 for a final benchmark. 30 is easier for a quick Colab run.
    N_TRIALS: int = 30

    # Same maximum wall-clock budget for CP-SAT, SA, and QI.
    TIME_BUDGET_SECONDS: float = 2.0

    # Use one worker for journal-style fairness: SA and QI are single-threaded
    # Python implementations, so multi-worker CP-SAT would receive extra CPU.
    NUM_WORKERS: int = 1

    # API synthesis campaign size.
    # More repeated products makes this closer to a real API campaign.
    # Each product appears in several batches, so product/campaign blocks matter.
    N_API_BATCHES: int = 32
    N_API_PRODUCTS: int = 8
    PRODUCTS_PER_CAMPAIGN_GROUP: int = 2
    N_SCENARIOS: int = 5

    # Same maximum candidate-evaluation cap for SA and QI.
    LOCAL_MAX_EVALS: int = 700

    # Same early stopping rule for SA and QI.
    # For paper-style fairness, this is set equal to LOCAL_MAX_EVALS so both
    # heuristic solvers are primarily limited by the same time/evaluation budget
    # rather than by algorithm-dependent sweep/chains semantics.
    MAX_NO_IMPROVEMENT_ROUNDS: int = 700

    RUSH_SEED: int = 1777

    # Sequence-dependent cleaning/changeover times.
    SAME_PRODUCT_CLEAN: int = 10
    SAME_GROUP_CLEAN: int = 90
    DIFFERENT_GROUP_CLEAN: int = 520

    # Serial-schedule generation rule.
    # A batch priority sequence is respected when several operations are close
    # in earliest start time; the window allows realistic pipelining.
    DISPATCH_WINDOW: int = 80

    # Same annealing schedule for SA and QI.
    TEMP_START: float = 350.0
    TEMP_END: float = 1.0

    # Objective weights. These are used by every method through the same evaluator.
    # Internal composite objective used by every method.
    # It is built only from the four reported business goals:
    # throughput, makespan, cleaning/changeover, and due-date performance.
    OBJECTIVE_SETUP_WEIGHT: float = 0.35
    OBJECTIVE_TARDINESS_WEIGHT: float = 0.018
    OBJECTIVE_LOST_THROUGHPUT_WEIGHT: float = 22.0
    OBJECTIVE_IDLE_WEIGHT: float = 0.0

    # CP-SAT nominal model objective weights.
    # CP-SAT is still finally evaluated by the same evaluator as every method.
    CPSAT_TARDINESS_WEIGHT: int = 1

    # MPO/TN-inspired QI parameters.
    # Bond dimension = number of retained candidate states after truncation.
    MPO_BOND_DIM: int = 10

    # Local physical states = candidate insertion/equipment states per API family block.
    MPO_LOCAL_STATES: int = 14

    # Number of tensor-network sweeps.
    MPO_SWEEPS: int = 5

    # Hamiltonian weights. These guide candidate ranking only.
    H_CLEAN_WEIGHT: float = 0.70
    H_DUE_WEIGHT: float = 220.0
    H_DEMAND_WEIGHT: float = 1.25
    H_LOAD_BALANCE_WEIGHT: float = 0.10
    H_SPEED_WEIGHT: float = 0.30
    H_FRAGMENT_WEIGHT: float = 45.0

    # Candidate-generation retry cap used by both local-search heuristics to
    # avoid counting exact no-op candidates as evaluated candidates.
    MAX_PROPOSAL_ATTEMPTS: int = 25


CFG = BenchmarkConfig()


# ============================================================
# 2. API SYNTHESIS CAMPAIGN STRUCTURE
# ============================================================

STAGES = ["SYNTHESIS", "PURIFICATION", "DRYING", "PACKAGING"]

EQUIPMENT_BY_STAGE = {
    "SYNTHESIS": ["R1", "R2", "R3"],       # reactors
    "PURIFICATION": ["P1", "P2"],          # purification skids/columns
    "DRYING": ["D1", "D2"],                # dryers
    "PACKAGING": ["L1", "L2"],             # packaging lines
}

STAGE_OF_EQUIPMENT = {
    equipment: stage
    for stage, equipment_list in EQUIPMENT_BY_STAGE.items()
    for equipment in equipment_list
}


def campaign_group(product):
    return int(product) // CFG.PRODUCTS_PER_CAMPAIGN_GROUP


def compatible_equipment(product: int, stage: str) -> List[str]:
    """
    Product-dependent equipment compatibility.

    This models the pharma reality that not every API can run on every reactor,
    purification train, dryer, or packaging line.
    """
    group = campaign_group(product)

    if stage == "SYNTHESIS":
        options = ["R1", "R2"]
        if product % 2 == 0 or group % 2 == 1:
            options.append("R3")
        return options

    if stage == "PURIFICATION":
        if product % 5 == 0:
            return ["P1"]
        if product % 5 == 1:
            return ["P2"]
        return ["P1", "P2"]

    if stage == "DRYING":
        if group % 3 == 0:
            return ["D1"]
        if group % 3 == 1:
            return ["D2"]
        return ["D1", "D2"]

    if stage == "PACKAGING":
        return ["L1", "L2"] if product % 4 != 0 else ["L1"]

    raise ValueError(f"Unknown stage: {stage}")


def setup_time(prev_product, next_product):
    """
    Sequence-dependent cleaning/changeover time.

    Same API product:
        small cleaning/changeover.
    Same campaign group:
        medium cleaning/changeover.
    Different campaign groups:
        large cleaning/changeover.
    """
    if prev_product is None:
        return 0

    if prev_product == next_product:
        return CFG.SAME_PRODUCT_CLEAN

    if campaign_group(prev_product) == campaign_group(next_product):
        return CFG.SAME_GROUP_CLEAN

    return CFG.DIFFERENT_GROUP_CLEAN


def stage_base_processing_time(stage: str, product: int, rng: random.Random) -> int:
    """
    Stage-specific nominal processing times.
    """
    complexity = 1 + (product % 4)

    if stage == "SYNTHESIS":
        return rng.randint(120, 260) + 20 * complexity
    if stage == "PURIFICATION":
        return rng.randint(80, 180) + 15 * complexity
    if stage == "DRYING":
        return rng.randint(70, 170) + 12 * complexity
    if stage == "PACKAGING":
        return rng.randint(45, 120) + 8 * complexity

    raise ValueError(f"Unknown stage: {stage}")


def equipment_speed_factor(equipment: str) -> float:
    """
    Some units are faster/slower. This creates a genuine assignment decision.
    """
    return {
        "R1": 1.00,
        "R2": 0.92,
        "R3": 1.12,
        "P1": 1.00,
        "P2": 0.90,
        "D1": 1.00,
        "D2": 0.94,
        "L1": 1.00,
        "L2": 0.88,
    }[equipment]


def generate_trial_case(trial_id: int):
    """
    Generate one API synthesis campaign scheduling instance.

    Each batch has:
        - id
        - API product
        - demand in kg
        - base raw-material release time
        - base due date
        - processing time for every compatible stage/equipment pair

    Several scenarios are generated for release times, due dates, and demand
    forecasts. The same scenario set is used by every method.
    """
    rng = random.Random(CFG.RUSH_SEED + trial_id)

    batches = []

    for j in range(CFG.N_API_BATCHES):
        product = (5 * j + 3 * trial_id + 2) % CFG.N_API_PRODUCTS
        demand_kg = rng.randint(12, 55)
        base_release = rng.randint(0, 220)

        processing = {}
        route_sum_fast = 0

        for stage in STAGES:
            base_p = stage_base_processing_time(stage, product, rng)
            processing[stage] = {}

            for eq in compatible_equipment(product, stage):
                p = int(round(base_p * equipment_speed_factor(eq)))
                processing[stage][eq] = max(1, p)

            route_sum_fast += min(processing[stage].values())

        # Due dates are tight enough to make throughput/tardiness meaningful.
        base_due = base_release + route_sum_fast + rng.randint(260, 620)

        batches.append(
            {
                "id": int(j),
                "product": int(product),
                "group": int(campaign_group(product)),
                "demand_kg": int(demand_kg),
                "base_release": int(base_release),
                "base_due": int(base_due),
                "processing": processing,
            }
        )

    scenarios = []

    for s in range(CFG.N_SCENARIOS):
        srng = random.Random(100000 + 1000 * trial_id + s)
        releases = []
        due_dates = []
        demands = []

        for batch in batches:
            product = batch["product"]
            group_shift = (campaign_group(product) * 35 + 17 * s) % 120
            release_jitter = srng.randint(0, 120)
            due_jitter = srng.randint(-120, 160)
            demand_multiplier = 0.80 + 0.10 * (s % 5) + srng.random() * 0.25

            release = batch["base_release"] + group_shift + release_jitter
            due = max(release + 200, batch["base_due"] + due_jitter)
            demand = max(1, int(round(batch["demand_kg"] * demand_multiplier)))

            releases.append(int(release))
            due_dates.append(int(due))
            demands.append(int(demand))

        scenarios.append(
            {
                "release": releases,
                "due": due_dates,
                "demand": demands,
            }
        )

    return batches, scenarios


# ============================================================
# 3. CAMPAIGN PLAN REPRESENTATION AND VALIDATION
# ============================================================


def is_valid_sequence(sequence: List[int], n: int) -> bool:
    return sorted(sequence) == list(range(n))


def copy_assignment(assignment: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {int(j): dict(stage_map) for j, stage_map in assignment.items()}


def copy_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sequence": list(plan["sequence"]),
        "assignment": copy_assignment(plan["assignment"]),
    }


def is_valid_assignment(batches, assignment: Dict[int, Dict[str, str]]) -> bool:
    n = len(batches)

    if set(assignment.keys()) != set(range(n)):
        return False

    for batch in batches:
        j = batch["id"]
        product = batch["product"]

        if set(assignment[j].keys()) != set(STAGES):
            return False

        for stage in STAGES:
            eq = assignment[j][stage]
            if eq not in compatible_equipment(product, stage):
                return False

    return True


def is_valid_plan(batches, plan: Dict[str, Any]) -> bool:
    n = len(batches)
    return (
        is_valid_sequence(plan["sequence"], n)
        and is_valid_assignment(batches, plan["assignment"])
    )


def initial_assignment(batches, trial_id: int) -> Dict[int, Dict[str, str]]:
    """
    Shared initial equipment assignment.

    It is intentionally simple and feasible. It is not a hidden optimized
    assignment. Every optimization method starts from this same plan.
    """
    assignment = {}

    for batch in batches:
        j = batch["id"]
        product = batch["product"]
        assignment[j] = {}

        for stage in STAGES:
            options = compatible_equipment(product, stage)
            # Deterministic round-robin among compatible units.
            index = (j + trial_id + len(stage)) % len(options)
            assignment[j][stage] = options[index]

    return assignment


def incumbent_mixed_campaign_plan(batches, trial_id: int) -> Dict[str, Any]:
    """
    Shared feasible incumbent.

    This models a manually mixed campaign where urgent batches are inserted
    across different products. It is feasible but can create many cleaning
    changes and idle gaps.
    """
    buckets = defaultdict(list)

    for batch in batches:
        buckets[batch["product"]].append(batch["id"])

    for product in buckets:
        buckets[product].sort(key=lambda j: (batches[j]["base_due"], batches[j]["base_release"]))

    products = sorted(buckets)
    shift = 5 + (trial_id % 7)
    products.sort(key=lambda p: ((p * shift) % 17, p))

    sequence = []

    while any(buckets[p] for p in products):
        for p in products:
            if buckets[p]:
                sequence.append(buckets[p].pop(0))

    assignment = initial_assignment(batches, trial_id)

    plan = {
        "sequence": sequence,
        "assignment": assignment,
    }

    assert is_valid_plan(batches, plan)
    return plan


def validate_trial_fairness(batches, scenarios, incumbent_plan):
    """
    Trial-level fairness audit.
    """
    n = len(batches)

    assert n == CFG.N_API_BATCHES, "Wrong number of API batches."
    assert len(scenarios) == CFG.N_SCENARIOS, "Wrong number of scenarios."
    assert is_valid_plan(batches, incumbent_plan), "Invalid incumbent plan."

    seen_ids = sorted(batch["id"] for batch in batches)
    assert seen_ids == list(range(n)), "Batch IDs must be 0,...,n-1."

    for batch in batches:
        assert 0 <= batch["product"] < CFG.N_API_PRODUCTS, "Invalid API product."
        assert batch["demand_kg"] > 0, "Demand must be positive."
        assert batch["base_release"] >= 0, "Release times must be nonnegative."
        assert batch["base_due"] > batch["base_release"], "Due date must follow release."

        for stage in STAGES:
            assert stage in batch["processing"], f"Missing stage {stage}."
            assert batch["processing"][stage], f"No compatible equipment for {stage}."

            for eq, p in batch["processing"][stage].items():
                assert eq in compatible_equipment(batch["product"], stage)
                assert p > 0

    for scenario_index, scenario in enumerate(scenarios):
        for key in ["release", "due", "demand"]:
            assert len(scenario[key]) == n, f"Scenario {scenario_index} has wrong {key} length."

        for j in range(n):
            assert scenario["release"][j] >= 0
            assert scenario["due"][j] > scenario["release"][j]
            assert scenario["demand"][j] > 0

    return True


# ============================================================
# 4. COMMON CAMPAIGN DECODER AND EVALUATOR
# ============================================================


def decode_one_scenario(batches, scenario: Dict[str, List[int]], plan: Dict[str, Any]) -> Dict[str, float]:
    """
    Decode a fixed campaign plan under one scenario.

    The plan consists of:
        - a batch priority sequence
        - an equipment assignment for each batch and stage

    The decoder uses a deterministic serial schedule-generation scheme:
        - stages respect SYNTHESIS -> PURIFICATION -> DRYING -> PACKAGING
        - equipment can process only one operation at a time
        - sequence-dependent cleaning/changeover is paid on each equipment
        - the batch priority sequence breaks ties within a dispatch window

    This same decoder is used for every method.
    """
    sequence = plan["sequence"]
    assignment = plan["assignment"]
    n = len(batches)

    assert is_valid_plan(batches, plan)

    priority = {job_id: pos for pos, job_id in enumerate(sequence)}

    equipment_available = {eq: 0 for eq in STAGE_OF_EQUIPMENT}
    equipment_last_product = {eq: None for eq in STAGE_OF_EQUIPMENT}
    equipment_has_been_used = {eq: False for eq in STAGE_OF_EQUIPMENT}

    next_stage_index = {j: 0 for j in range(n)}
    previous_stage_finish = {j: scenario["release"][j] for j in range(n)}
    completion_time = {j: None for j in range(n)}

    total_cleaning = 0.0
    total_idle = 0.0
    scheduled_operations = 0
    total_operations = n * len(STAGES)

    while scheduled_operations < total_operations:
        candidates = []

        for j in range(n):
            stage_idx = next_stage_index[j]
            if stage_idx >= len(STAGES):
                continue

            stage = STAGES[stage_idx]
            eq = assignment[j][stage]
            product = batches[j]["product"]

            clean = setup_time(equipment_last_product[eq], product)
            earliest_start = max(
                previous_stage_finish[j],
                equipment_available[eq] + clean,
            )

            candidates.append(
                {
                    "batch": j,
                    "stage": stage,
                    "stage_idx": stage_idx,
                    "equipment": eq,
                    "clean": clean,
                    "earliest_start": earliest_start,
                    "priority": priority[j],
                }
            )

        if not candidates:
            raise RuntimeError("No schedulable operations found.")

        min_start = min(c["earliest_start"] for c in candidates)

        # Respect the campaign priority sequence when operations are close in time.
        eligible = [
            c for c in candidates
            if c["earliest_start"] <= min_start + CFG.DISPATCH_WINDOW
        ]

        chosen = min(
            eligible,
            key=lambda c: (c["priority"], c["earliest_start"], c["stage_idx"], c["batch"]),
        )

        j = chosen["batch"]
        stage = chosen["stage"]
        eq = chosen["equipment"]
        clean = chosen["clean"]
        start = int(chosen["earliest_start"])
        p = int(batches[j]["processing"][stage][eq])
        finish = start + p

        if equipment_has_been_used[eq]:
            idle_gap = max(0, start - (equipment_available[eq] + clean))
            total_idle += idle_gap

        total_cleaning += clean
        equipment_available[eq] = finish
        equipment_last_product[eq] = batches[j]["product"]
        equipment_has_been_used[eq] = True

        previous_stage_finish[j] = finish
        next_stage_index[j] += 1

        if next_stage_index[j] == len(STAGES):
            completion_time[j] = finish

        scheduled_operations += 1

    makespan = max(completion_time.values())
    weighted_tardiness = 0.0
    on_time_throughput = 0.0
    total_demand = 0.0
    due_hits = 0

    for j in range(n):
        demand = scenario["demand"][j]
        due = scenario["due"][j]
        completion = completion_time[j]
        tardiness = max(0, completion - due)

        weighted_tardiness += demand * tardiness
        total_demand += demand

        if completion <= due:
            on_time_throughput += demand
            due_hits += 1

    on_time_throughput_rate = on_time_throughput / max(1.0, total_demand)
    due_date_hit_rate = due_hits / max(1, n)
    throughput_kg_per_time = total_demand / max(1.0, makespan)

    # Shared internal objective.  This is not reported as a comparison metric;
    # it is only the common scalar score used by CP-SAT, SA, and QI to compare
    # candidate plans.  It uses only the four requested business goals.
    objective = (
        makespan
        + CFG.OBJECTIVE_SETUP_WEIGHT * total_cleaning
        + CFG.OBJECTIVE_TARDINESS_WEIGHT * weighted_tardiness
        + CFG.OBJECTIVE_LOST_THROUGHPUT_WEIGHT * (total_demand - on_time_throughput)
    )

    return {
        "Objective": float(objective),
        "ThroughputKgPerTime": float(throughput_kg_per_time),
        "Makespan": float(makespan),
        "CleaningChangeoverTime": float(total_cleaning),
        "DueDateHitRate": float(due_date_hit_rate),
        "WeightedTardiness": float(weighted_tardiness),
        "OnTimeThroughputKg": float(on_time_throughput),
        "TotalDemandKg": float(total_demand),
        "OnTimeThroughputRate": float(on_time_throughput_rate),
        "EquipmentIdleTime": float(total_idle),
    }


def evaluate_campaign_plan(batches, scenarios, plan: Dict[str, Any]) -> Dict[str, float]:
    """
    Main final evaluator used by every method.
    """
    scenario_metrics = [
        decode_one_scenario(batches, scenario, plan)
        for scenario in scenarios
    ]

    keys = list(scenario_metrics[0].keys())
    out = {}

    for key in keys:
        out[key] = float(np.mean([m[key] for m in scenario_metrics]))

    return out


def plan_objective(batches, scenarios, plan: Dict[str, Any]) -> float:
    return evaluate_campaign_plan(batches, scenarios, plan)["Objective"]


def mean_cleaning_of_plan(batches, scenarios, plan: Dict[str, Any]) -> float:
    return evaluate_campaign_plan(batches, scenarios, plan)["CleaningChangeoverTime"]


def plan_signature(plan: Dict[str, Any]) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, str, str], ...]]:
    """
    Deterministic plan signature used only for no-op/duplicate checks.

    The signature is independent of Python dictionary insertion order because
    batch ids and stages are traversed in canonical order.
    """
    return (
        tuple(plan["sequence"]),
        tuple(
            (j, stage, plan["assignment"][j][stage])
            for j in range(len(plan["sequence"]))
            for stage in STAGES
        ),
    )


# ============================================================
# 5. SHARED ANNEALING UTILITY
# ============================================================


def annealing_temperature(evaluated: int) -> float:
    """
    Same temperature schedule for SA and QI.
    """
    progress = evaluated / max(1, CFG.LOCAL_MAX_EVALS)

    return CFG.TEMP_START * (
        (CFG.TEMP_END / CFG.TEMP_START) ** progress
    )


def accept_move(candidate_score: float, current_score: float, temperature: float, rng: random.Random) -> bool:
    """
    Same acceptance rule for SA and QI.
    """
    if candidate_score < current_score:
        return True

    delta = candidate_score - current_score
    probability = math.exp(-delta / max(temperature, 1e-9))

    return rng.random() < probability


# ============================================================
# 6. CLASSICAL SIMULATED ANNEALING
# ============================================================


def random_valid_equipment_change(batches, plan: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Return a valid equipment reassignment that changes the current plan when
    such a reassignment exists.

    This avoids a fairness problem where SA could spend a counted objective
    evaluation on a no-op equipment move for a batch-stage with only one
    compatible equipment option, while QI skipped duplicate states.
    """
    candidate = copy_plan(plan)

    movable_pairs = []
    for batch in batches:
        j = batch["id"]
        product = batch["product"]
        for stage in STAGES:
            options = compatible_equipment(product, stage)
            current_eq = candidate["assignment"][j][stage]
            alternatives = [eq for eq in options if eq != current_eq]
            if alternatives:
                movable_pairs.append((j, stage, alternatives))

    if not movable_pairs:
        return candidate

    j, stage, alternatives = rng.choice(movable_pairs)
    candidate["assignment"][j][stage] = rng.choice(alternatives)
    return candidate


def structured_campaign_candidate_for_sa(
    batches,
    scenarios,
    plan: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Classical SA access to the same campaign-aware neighborhood family used by QI.

    This is included for benchmark fairness. QI should not be the only method
    that can make product-block, campaign-group, boundary-repair, critical-batch,
    or structured equipment moves. SA samples one candidate from the same public
    local-state families, while QI uses the tensor-network bond-state expansion
    and Hamiltonian truncation to organize those states.
    """
    selector = rng.random()
    states = []

    products = sorted({batch["product"] for batch in batches})
    groups = sorted({batch["group"] for batch in batches})

    if selector < 0.30 and products:
        product = rng.choice(products)
        states = local_states_for_api_product_block(
            batches, scenarios, plan, product, max(4, CFG.MPO_LOCAL_STATES // 2)
        )
    elif selector < 0.55 and groups:
        group = rng.choice(groups)
        states = local_states_for_campaign_group_block(
            batches, scenarios, plan, group, max(4, CFG.MPO_LOCAL_STATES // 2)
        )
    elif selector < 0.75:
        boundaries = high_cleaning_boundaries(
            batches, plan, max_boundaries=min(4, len(batches) - 1)
        )
        if boundaries:
            left_batch, right_batch, _clean = rng.choice(boundaries)
            states = local_states_for_boundary_repair(
                batches, scenarios, plan, left_batch, right_batch, max(3, CFG.MPO_LOCAL_STATES // 3)
            )
    else:
        critical = critical_batches_for_qi(
            batches, scenarios, plan, max_batches=min(4, len(batches))
        )
        if critical:
            batch_id = rng.choice(critical)
            states = local_states_for_critical_batch_site(
                batches, scenarios, plan, batch_id, max(3, CFG.MPO_LOCAL_STATES // 4)
            )

    if not states:
        return copy_plan(plan)

    # SA samples from the generated local states rather than taking the best
    # Hamiltonian-ranked state. This keeps the baseline classical and avoids
    # giving SA the TN truncation rule.
    candidate_plan, _candidate_h = rng.choice(states)
    return copy_plan(candidate_plan)


def classical_structured_simulated_annealing(
    batches,
    scenarios,
    incumbent_plan: Dict[str, Any],
    time_budget: float,
    trial_id: int,
):
    """
    Classical structured simulated annealing for API campaign scheduling.

    Fairness:
        - starts from the same incumbent as QI
        - same objective
        - same release/due/demand scenarios
        - same time budget
        - same candidate-evaluation cap as QI
        - same no-improvement stopping rule as QI
        - same annealing temperature schedule as QI
        - same acceptance rule as QI

    Moves:
        - swap two batches in the campaign priority sequence
        - insert one batch elsewhere
        - reverse an interval
        - change one batch-stage equipment assignment
        - randomly sample from the same campaign-aware block/equipment
          neighborhood families available to QI
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    rng = random.Random(30000 + trial_id)

    current = copy_plan(incumbent_plan)
    current_metrics = evaluate_campaign_plan(batches, scenarios, current)
    current_score = current_metrics["Objective"]

    best_plan = copy_plan(current)
    best_metrics = dict(current_metrics)
    best_score = current_score

    evaluated = 1
    no_improvement_rounds = 0

    while (
        time.perf_counter() < deadline
        and evaluated < CFG.LOCAL_MAX_EVALS
        and no_improvement_rounds < CFG.MAX_NO_IMPROVEMENT_ROUNDS
    ):
        current_sig = plan_signature(current)
        candidate = None

        for _attempt in range(CFG.MAX_PROPOSAL_ATTEMPTS):
            proposal = copy_plan(current)
            move = rng.random()

            if move < 0.20:
                i, j = rng.sample(range(len(proposal["sequence"])), 2)
                proposal["sequence"][i], proposal["sequence"][j] = (
                    proposal["sequence"][j],
                    proposal["sequence"][i],
                )

            elif move < 0.40:
                i, j = rng.sample(range(len(proposal["sequence"])), 2)
                item = proposal["sequence"].pop(i)
                proposal["sequence"].insert(j, item)

            elif move < 0.55:
                i, j = sorted(rng.sample(range(len(proposal["sequence"])), 2))
                proposal["sequence"][i:j + 1] = reversed(proposal["sequence"][i:j + 1])

            elif move < 0.70:
                proposal = random_valid_equipment_change(batches, proposal, rng)

            else:
                proposal = structured_campaign_candidate_for_sa(
                    batches, scenarios, proposal, rng
                )

            assert is_valid_plan(batches, proposal)

            if plan_signature(proposal) != current_sig:
                candidate = proposal
                break

        # If all attempts returned a duplicate, do not spend a counted full
        # objective evaluation. QI uses the same principle for duplicate states.
        if candidate is None:
            continue

        candidate_metrics = evaluate_campaign_plan(batches, scenarios, candidate)
        candidate_score = candidate_metrics["Objective"]
        evaluated += 1

        temperature = annealing_temperature(evaluated)

        if accept_move(candidate_score, current_score, temperature, rng):
            current = candidate
            current_score = candidate_score

            if candidate_score < best_score:
                best_plan = copy_plan(candidate)
                best_score = candidate_score
                best_metrics = dict(candidate_metrics)
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
        f"final_no_improvement_count={no_improvement_rounds}; "
        f"move_type=sequence_swap_insert_reverse_plus_equipment_reassignment_plus_shared_campaign_neighborhoods"
    )

    return best_plan, best_metrics, runtime, info


# ============================================================
# 7. QI MPO-HAMILTONIAN TENSOR-NETWORK CAMPAIGN REPAIR
# ============================================================


_AVG_SCENARIO_VECTOR_CACHE = {}

def average_scenario_vector(scenarios, key: str) -> Dict[int, float]:
    """
    Average scenario data for one quantity.

    The cache key is based on scenario content rather than object identity. This
    avoids a subtle reproducibility bug: Python can reuse object ids after a
    trial is garbage-collected, which could otherwise return averages from a
    different trial. The helper uses only public scenario data available to all
    methods.
    """
    cache_key = (key, tuple(tuple(scenario[key]) for scenario in scenarios))
    cached = _AVG_SCENARIO_VECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    n = len(scenarios[0][key])
    out = {
        j: float(np.mean([scenario[key][j] for scenario in scenarios]))
        for j in range(n)
    }
    _AVG_SCENARIO_VECTOR_CACHE[cache_key] = out
    return out


def campaign_fragmentation_energy(batches, plan: Dict[str, Any]) -> float:
    """
    Penalize splitting the same API product into separated campaign fragments.
    """
    fragments = defaultdict(int)
    previous_product = None

    for j in plan["sequence"]:
        product = batches[j]["product"]
        if product != previous_product:
            fragments[product] += 1
        previous_product = product

    penalty = 0.0
    for _, count in fragments.items():
        if count > 1:
            penalty += count - 1

    return float(penalty)


def sequence_cleaning_surrogate(batches, plan: Dict[str, Any]) -> float:
    """
    Cheap nearest-neighbor MPO-like cleaning surrogate.

    It approximates the cleaning/changeover burden induced by the priority
    sequence and stage-wise equipment assignments. Unlike the previous version,
    this does not call the full final schedule decoder. That lets QI spend the
    same time/evaluation budget on exploring more useful local states.
    """
    sequence = plan["sequence"]
    assignment = plan["assignment"]
    total = 0.0

    for stage in STAGES:
        for eq in EQUIPMENT_BY_STAGE[stage]:
            previous_product = None
            for j in sequence:
                if assignment[j][stage] != eq:
                    continue
                product = batches[j]["product"]
                total += setup_time(previous_product, product)
                previous_product = product

    return float(total)


def mean_due_date_pressure(batches, scenarios, plan: Dict[str, Any]) -> float:
    """
    One-site due-date pressure.

    Early-due and high-demand batches are penalized if they are placed late in
    the priority sequence. This is a structural surrogate, not the final score.
    """
    n = len(batches)
    position = {j: pos for pos, j in enumerate(plan["sequence"])}

    avg_due = average_scenario_vector(scenarios, "due")
    avg_demand = average_scenario_vector(scenarios, "demand")
    avg_release = average_scenario_vector(scenarios, "release")

    due_values = list(avg_due.values())
    release_values = list(avg_release.values())
    demand_values = list(avg_demand.values())

    d_min = min(due_values)
    d_max = max(due_values)
    r_min = min(release_values)
    r_max = max(release_values)
    demand_max = max(1.0, max(demand_values))

    due_denom = max(1.0, d_max - d_min)
    release_denom = max(1.0, r_max - r_min)

    pressure = 0.0

    for j in range(n):
        early_due_urgency = 1.0 - ((avg_due[j] - d_min) / due_denom)
        early_release_urgency = 1.0 - ((avg_release[j] - r_min) / release_denom)
        demand_weight = avg_demand[j] / demand_max
        normalized_position = position[j] / max(1, n - 1)

        pressure += (
            0.70 * early_due_urgency
            + 0.30 * early_release_urgency
        ) * demand_weight * normalized_position

    return float(pressure)


def equipment_load_balance_energy(batches, plan: Dict[str, Any]) -> float:
    """
    Penalize overloaded equipment assignments within each stage.
    """
    stage_loads = {stage: defaultdict(float) for stage in STAGES}

    for batch in batches:
        j = batch["id"]
        for stage in STAGES:
            eq = plan["assignment"][j][stage]
            stage_loads[stage][eq] += batch["processing"][stage][eq]

    energy = 0.0

    for stage in STAGES:
        loads = [stage_loads[stage][eq] for eq in EQUIPMENT_BY_STAGE[stage]]
        if len(loads) <= 1:
            continue
        energy += float(np.std(loads))

    return float(energy)


def equipment_speed_loss_energy(batches, scenarios, plan: Dict[str, Any]) -> float:
    """
    Penalize assigning urgent/high-demand batches to much slower compatible units.

    This improves the equipment component of the QI local states without giving
    QI extra information. It uses only compatible processing times and demand/due
    scenario data already used by the common evaluator.
    """
    avg_due = average_scenario_vector(scenarios, "due")
    avg_demand = average_scenario_vector(scenarios, "demand")

    due_values = list(avg_due.values())
    d_min = min(due_values)
    d_max = max(due_values)
    due_denom = max(1.0, d_max - d_min)
    demand_max = max(1.0, max(avg_demand.values()))

    energy = 0.0

    for batch in batches:
        j = batch["id"]
        urgency = 1.0 - ((avg_due[j] - d_min) / due_denom)
        demand_weight = avg_demand[j] / demand_max

        for stage in STAGES:
            assigned_eq = plan["assignment"][j][stage]
            assigned_p = batches[j]["processing"][stage][assigned_eq]
            best_p = min(batches[j]["processing"][stage].values())
            energy += urgency * demand_weight * max(0, assigned_p - best_p)

    return float(energy)


def throughput_pressure_energy(batches, scenarios, plan: Dict[str, Any]) -> float:
    """
    Cheap demand-throughput pressure.

    The previous implementation called evaluate_campaign_plan inside the
    Hamiltonian. This version estimates lost-throughput risk from due-date
    slack, demand, route duration, equipment speed loss, and sequence position.
    """
    n = len(batches)
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")
    position = {j: pos for pos, j in enumerate(plan["sequence"])}

    pressure = 0.0

    for batch in batches:
        j = batch["id"]
        assigned_route = sum(
            batches[j]["processing"][stage][plan["assignment"][j][stage]]
            for stage in STAGES
        )
        fastest_route = sum(
            min(batches[j]["processing"][stage].values())
            for stage in STAGES
        )
        slack = avg_due[j] - avg_release[j] - fastest_route
        normalized_lateness_risk = max(0.0, assigned_route - slack) / max(1.0, fastest_route)
        normalized_position = position[j] / max(1, n - 1)
        pressure += avg_demand[j] * normalized_lateness_risk * (1.0 + normalized_position)

    return float(pressure)


def mpo_hamiltonian_energy(batches, scenarios, plan: Dict[str, Any]) -> float:
    """
    MPO/Hamiltonian surrogate energy.

    H(plan)
        = w_clean    * H_clean_surrogate
        + w_due      * H_due_pressure
        + w_demand   * H_throughput_pressure
        + w_load     * H_load_balance
        + w_speed    * H_equipment_speed_loss
        + w_fragment * H_fragment

    Important:
        This Hamiltonian is used only to guide and truncate QI candidate states.
        Final benchmark scores are computed by evaluate_campaign_plan, exactly
        the same as for all other methods.
    """
    h_clean = sequence_cleaning_surrogate(batches, plan)
    h_due = mean_due_date_pressure(batches, scenarios, plan)
    h_demand = throughput_pressure_energy(batches, scenarios, plan)
    h_load = equipment_load_balance_energy(batches, plan)
    h_speed = equipment_speed_loss_energy(batches, scenarios, plan)
    h_fragment = campaign_fragmentation_energy(batches, plan)

    return float(
        CFG.H_CLEAN_WEIGHT * h_clean
        + CFG.H_DUE_WEIGHT * h_due
        + CFG.H_DEMAND_WEIGHT * h_demand
        + CFG.H_LOAD_BALANCE_WEIGHT * h_load
        + CFG.H_SPEED_WEIGHT * h_speed
        + CFG.H_FRAGMENT_WEIGHT * h_fragment
    )


def product_boundary_penalty(batches, plan: Dict[str, Any], product: int) -> float:
    """
    Cleaning penalty around one API product block in the current sequence.
    """
    positions = [
        i for i, j in enumerate(plan["sequence"])
        if batches[j]["product"] == product
    ]

    if not positions:
        return 0.0

    first = positions[0]
    last = positions[-1]
    sequence = plan["sequence"]

    penalty = 0.0

    if first > 0:
        penalty += setup_time(
            batches[sequence[first - 1]]["product"],
            product,
        )

    if last + 1 < len(sequence):
        penalty += setup_time(
            product,
            batches[sequence[last + 1]]["product"],
        )

    return float(penalty)


def product_urgency_score(batches, scenarios, plan: Dict[str, Any], product: int) -> float:
    """
    Current-state priority score for deciding sweep order.
    """
    block = [j for j in plan["sequence"] if batches[j]["product"] == product]
    if not block:
        return 0.0

    avg_due = average_scenario_vector(scenarios, "due")
    avg_demand = average_scenario_vector(scenarios, "demand")
    position = {j: pos for pos, j in enumerate(plan["sequence"])}
    n = len(plan["sequence"])

    due_values = list(avg_due.values())
    d_min = min(due_values)
    d_max = max(due_values)
    due_denom = max(1.0, d_max - d_min)
    demand_max = max(1.0, max(avg_demand.values()))

    score = product_boundary_penalty(batches, plan, product)

    for j in block:
        urgency = 1.0 - ((avg_due[j] - d_min) / due_denom)
        demand_weight = avg_demand[j] / demand_max
        late_position = position[j] / max(1, n - 1)
        score += 180.0 * urgency * demand_weight * late_position

    return float(score)


def fastest_assignment_variant_for_block(batches, plan: Dict[str, Any], block: List[int]) -> Dict[int, Dict[str, str]]:
    assignment = copy_assignment(plan["assignment"])

    for j in block:
        product = batches[j]["product"]
        for stage in STAGES:
            options = compatible_equipment(product, stage)
            best_eq = min(options, key=lambda eq: batches[j]["processing"][stage][eq])
            assignment[j][stage] = best_eq

    return assignment


def load_balanced_assignment_variant_for_block(batches, plan: Dict[str, Any], block: List[int]) -> Dict[int, Dict[str, str]]:
    assignment = copy_assignment(plan["assignment"])

    current_load = {eq: 0.0 for eq in STAGE_OF_EQUIPMENT}

    for batch in batches:
        j = batch["id"]
        if j in block:
            continue
        for stage in STAGES:
            eq = assignment[j][stage]
            current_load[eq] += batches[j]["processing"][stage][eq]

    for j in block:
        product = batches[j]["product"]
        for stage in STAGES:
            options = compatible_equipment(product, stage)
            best_eq = min(
                options,
                key=lambda eq: (
                    current_load[eq] + batches[j]["processing"][stage][eq],
                    batches[j]["processing"][stage][eq],
                ),
            )
            assignment[j][stage] = best_eq
            current_load[best_eq] += batches[j]["processing"][stage][best_eq]

    return assignment


def campaign_consistent_assignment_variant_for_block(batches, plan: Dict[str, Any], block: List[int]) -> Dict[int, Dict[str, str]]:
    """
    Try to assign the whole product block consistently within each stage when
    possible. This models campaign operation on the same reactor/train/dryer/line.
    """
    assignment = copy_assignment(plan["assignment"])

    for stage in STAGES:
        common_options = None
        for j in block:
            product = batches[j]["product"]
            options = set(compatible_equipment(product, stage))
            common_options = options if common_options is None else common_options & options

        if common_options:
            best_eq = min(
                common_options,
                key=lambda eq: sum(batches[j]["processing"][stage][eq] for j in block),
            )
            for j in block:
                assignment[j][stage] = best_eq

    return assignment


def bottleneck_aware_assignment_variant_for_block(
    batches,
    scenarios,
    plan: Dict[str, Any],
    block: List[int],
) -> Dict[int, Dict[str, str]]:
    """
    Assign the block using a bottleneck-aware local rule.

    This is still a local state, not a solve-ahead. It uses current equipment
    load, the cleaning transition on that equipment, and public due/demand data.
    """
    assignment = copy_assignment(plan["assignment"])
    avg_due = average_scenario_vector(scenarios, "due")
    avg_demand = average_scenario_vector(scenarios, "demand")

    current_load = {eq: 0.0 for eq in STAGE_OF_EQUIPMENT}
    last_product = {eq: None for eq in STAGE_OF_EQUIPMENT}

    block_set = set(block)
    for j in plan["sequence"]:
        if j in block_set:
            continue
        product = batches[j]["product"]
        for stage in STAGES:
            eq = assignment[j][stage]
            current_load[eq] += batches[j]["processing"][stage][eq]
            last_product[eq] = product

    ordered_block = sorted(
        block,
        key=lambda j: (avg_due[j], -avg_demand[j], batches[j]["base_release"], j),
    )

    for j in ordered_block:
        product = batches[j]["product"]
        for stage in STAGES:
            options = compatible_equipment(product, stage)
            best_eq = min(
                options,
                key=lambda eq: (
                    current_load[eq]
                    + setup_time(last_product[eq], product)
                    + batches[j]["processing"][stage][eq],
                    batches[j]["processing"][stage][eq],
                    eq,
                ),
            )
            assignment[j][stage] = best_eq
            current_load[best_eq] += (
                setup_time(last_product[best_eq], product)
                + batches[j]["processing"][stage][best_eq]
            )
            last_product[best_eq] = product

    return assignment


def block_order_variants(batches, scenarios, block: List[int]) -> List[List[int]]:
    """
    Candidate local orderings inside one API product block.

    The previous QI version moved the whole product block but preserved its
    internal order. For API campaign scheduling, internal order matters because
    batches of the same API can have different due dates and demand forecasts.
    """
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    variants = []
    base = list(block)
    variants.append(base)

    due_order = sorted(block, key=lambda j: (avg_due[j], -avg_demand[j], avg_release[j], j))
    release_order = sorted(block, key=lambda j: (avg_release[j], avg_due[j], -avg_demand[j], j))
    demand_order = sorted(block, key=lambda j: (-avg_demand[j], avg_due[j], avg_release[j], j))

    for candidate in [due_order, release_order, demand_order]:
        if candidate not in variants:
            variants.append(candidate)

    return variants


def due_date_candidate_positions(batches, scenarios, rest: List[int], block: List[int]) -> List[int]:
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    block_due = float(np.average(
        [avg_due[j] for j in block],
        weights=[max(1.0, avg_demand[j]) for j in block],
    ))
    block_release = float(np.mean([avg_release[j] for j in block]))

    positions = {0, len(rest)}

    # Position in the current sequence where the block starts to look urgent.
    for pos, job_id in enumerate(rest):
        if avg_due[job_id] >= block_due:
            positions.add(pos)
            break

    # Position in the current sequence based on release readiness.
    for pos, job_id in enumerate(rest):
        if avg_release[job_id] >= block_release:
            positions.add(pos)
            break

    # Due/release quantile anchors. These are only anchors; final scoring still
    # uses the common evaluator after a candidate is formed.
    due_sorted_count = sum(1 for j in rest if avg_due[j] <= block_due)
    release_sorted_count = sum(1 for j in rest if avg_release[j] <= block_release)
    positions.add(due_sorted_count)
    positions.add(release_sorted_count)

    # Add a few nearby positions around those anchors.
    for anchor in list(positions):
        for delta in [-2, -1, 1, 2]:
            positions.add(anchor + delta)

    return sorted(p for p in positions if 0 <= p <= len(rest))


def movable_units_for_product(batches, scenarios, sequence: List[int], product: int) -> List[List[int]]:
    """
    Product-level local units for the QI tensor site.

    The primary state is the full product campaign block. When due dates are
    tight, the local site is also allowed to move the most urgent sub-block.
    This keeps the campaign interpretation but avoids forcing every batch of
    the same API to move together when one urgent batch drives tardiness.
    """
    full_block = [j for j in sequence if batches[j]["product"] == product]
    if not full_block:
        return []

    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    due_order = sorted(full_block, key=lambda j: (avg_due[j], -avg_demand[j], avg_release[j], j))

    units = []
    for candidate in [full_block, due_order]:
        if candidate not in units:
            units.append(candidate)

    if len(due_order) >= 2:
        urgent_one = [due_order[0]]
        if urgent_one not in units:
            units.append(urgent_one)

    if len(due_order) >= 4:
        urgent_half = due_order[: max(2, len(due_order) // 2)]
        if urgent_half not in units:
            units.append(urgent_half)

    return units




def rank_candidate_positions(
    batches,
    scenarios,
    rest: List[int],
    move_unit: List[int],
    product: int,
    positions: List[int],
    max_positions: int,
) -> List[int]:
    """
    Keep only the most promising local insertion positions before building
    equipment/order variants. This avoids wasting the equal QI time budget on
    thousands of surrogate-only states.
    """
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    move_due = float(np.average(
        [avg_due[j] for j in move_unit],
        weights=[max(1.0, avg_demand[j]) for j in move_unit],
    ))
    move_release = float(np.mean([avg_release[j] for j in move_unit]))

    def score_position(pos: int):
        left_product = batches[rest[pos - 1]]["product"] if pos > 0 else None
        right_product = batches[rest[pos]]["product"] if pos < len(rest) else None
        left_cost = setup_time(left_product, product)
        right_cost = 0 if right_product is None else setup_time(product, right_product)

        due_mismatch = 0.0
        release_mismatch = 0.0
        if pos > 0:
            due_mismatch += max(0.0, avg_due[rest[pos - 1]] - move_due) / 100.0
            release_mismatch += max(0.0, avg_release[rest[pos - 1]] - move_release) / 100.0
        if pos < len(rest):
            due_mismatch += max(0.0, move_due - avg_due[rest[pos]]) / 100.0
            release_mismatch += max(0.0, move_release - avg_release[rest[pos]]) / 100.0

        group_bonus = 0.0
        if left_product is not None and campaign_group(left_product) == campaign_group(product):
            group_bonus -= 25.0
        if right_product is not None and campaign_group(right_product) == campaign_group(product):
            group_bonus -= 25.0

        return (left_cost + right_cost + 80.0 * due_mismatch + 20.0 * release_mismatch + group_bonus, pos)

    unique_positions = sorted(set(positions))
    ranked = sorted(unique_positions, key=score_position)
    return ranked[:max_positions]
def group_urgency_score(batches, scenarios, plan: Dict[str, Any], group: int) -> float:
    """Priority score for a whole campaign group contraction."""
    products = {batch["product"] for batch in batches if batch["group"] == group}
    return float(sum(product_urgency_score(batches, scenarios, plan, p) for p in products))


def block_order_variants_general(batches, scenarios, block: List[int]) -> List[List[int]]:
    """Order variants for a mixed product/campaign-group block."""
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    variants = []

    by_product_then_due = sorted(
        block,
        key=lambda j: (
            campaign_group(batches[j]["product"]),
            batches[j]["product"],
            avg_due[j],
            avg_release[j],
            -avg_demand[j],
            j,
        ),
    )
    by_due_then_product = sorted(
        block,
        key=lambda j: (
            avg_due[j],
            campaign_group(batches[j]["product"]),
            batches[j]["product"],
            -avg_demand[j],
            avg_release[j],
            j,
        ),
    )
    by_group_due = sorted(
        block,
        key=lambda j: (
            campaign_group(batches[j]["product"]),
            avg_due[j],
            -avg_demand[j],
            avg_release[j],
            j,
        ),
    )

    for candidate in [list(block), by_product_then_due, by_due_then_product, by_group_due]:
        if candidate not in variants:
            variants.append(candidate)

    return variants


def local_states_for_campaign_group_block(
    batches,
    scenarios,
    plan: Dict[str, Any],
    group: int,
    max_states: int,
):
    """
    Two-site / block-contraction TN move for one API campaign group.

    This is the main QI improvement in this version.  Product-level moves can be
    too small for API synthesis campaigns, because changeover costs are often
    reduced only when an entire related campaign group is compacted.  The move
    is still fair: it uses only public product-group labels, due dates, demand,
    equipment compatibility, the same final evaluator, and the same candidate
    cap/time budget.
    """
    sequence = plan["sequence"]
    full_block = [j for j in sequence if batches[j]["group"] == group]
    if not full_block:
        return []

    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    # Full group block plus the most urgent half of that group.
    due_order = sorted(full_block, key=lambda j: (avg_due[j], -avg_demand[j], avg_release[j], j))
    move_units = [full_block]
    urgent_half = due_order[: max(2, len(due_order) // 2)]
    if urgent_half != full_block and urgent_half not in move_units:
        move_units.append(urgent_half)

    candidates = []
    seen_signatures = set()

    for move_unit in move_units[:2]:
        move_set = set(move_unit)
        rest = [j for j in sequence if j not in move_set]

        candidate_positions = set(due_date_candidate_positions(batches, scenarios, rest, move_unit))

        for pos, job_id in enumerate(rest):
            if batches[job_id]["group"] == group:
                candidate_positions.add(pos)
                candidate_positions.add(pos + 1)

        # Add quantile anchors so the group can move far enough to change the
        # campaign structure, but still through local TN states.
        for frac in [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]:
            candidate_positions.add(int(round(frac * len(rest))))

        clean_positions = sorted(p for p in candidate_positions if 0 <= p <= len(rest))
        clean_positions = rank_candidate_positions(
            batches,
            scenarios,
            rest,
            move_unit,
            batches[move_unit[0]]["product"],
            clean_positions,
            max_positions=max(8, max_states),
        )

        block_variants = block_order_variants_general(batches, scenarios, move_unit)[:4]
        assignment_variants = [
            copy_assignment(plan["assignment"]),
            fastest_assignment_variant_for_block(batches, plan, move_unit),
            load_balanced_assignment_variant_for_block(batches, plan, move_unit),
            campaign_consistent_assignment_variant_for_block(batches, plan, move_unit),
            bottleneck_aware_assignment_variant_for_block(batches, scenarios, plan, move_unit),
        ]

        for p in clean_positions:
            for ordered_block in block_variants:
                new_sequence = rest[:p] + ordered_block + rest[p:]
                for assignment in assignment_variants:
                    candidate = {
                        "sequence": list(new_sequence),
                        "assignment": copy_assignment(assignment),
                    }
                    if not is_valid_plan(batches, candidate):
                        continue
                    signature = (
                        tuple(candidate["sequence"]),
                        tuple(
                            (j, stage, candidate["assignment"][j][stage])
                            for j in range(len(batches))
                            for stage in STAGES
                        ),
                    )
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    candidates.append((candidate, mpo_hamiltonian_energy(batches, scenarios, candidate)))

    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[:max_states]


def local_states_for_api_product_block(
    batches,
    scenarios,
    plan: Dict[str, Any],
    product: int,
    max_states: int,
):
    """
    Local physical states for one API product/campaign site.

    Tensor-network interpretation:
        - Each API product is a tensor-network site.
        - A local state can move the full product campaign block or, when due
          dates justify it, an urgent sub-block of that product.
        - A local state also includes an internal block order and feasible
          equipment-assignment pattern.
        - The Hamiltonian ranks local states and the bond dimension truncates
          the retained candidates.

    This is not a precomputed solution. It is a local candidate generator using
    the current plan and shared problem data.
    """
    sequence = plan["sequence"]
    move_units = movable_units_for_product(batches, scenarios, sequence, product)[:3]

    if not move_units:
        return []

    candidates = []
    seen_signatures = set()

    for move_unit in move_units:
        move_set = set(move_unit)
        rest = [j for j in sequence if j not in move_set]

        candidate_positions = set(due_date_candidate_positions(batches, scenarios, rest, move_unit))

        # Place near the same campaign group to reduce changeovers.
        for pos, job_id in enumerate(rest):
            if campaign_group(batches[job_id]["product"]) == campaign_group(product):
                candidate_positions.add(pos)
                candidate_positions.add(pos + 1)

        # Place near products with low boundary cleaning cost.
        for pos in range(len(rest) + 1):
            left_product = batches[rest[pos - 1]]["product"] if pos > 0 else None
            right_product = batches[rest[pos]]["product"] if pos < len(rest) else None
            left_cost = setup_time(left_product, product)
            right_cost = 0 if right_product is None else setup_time(product, right_product)
            boundary_cost = left_cost + right_cost
            if boundary_cost <= CFG.SAME_GROUP_CLEAN + CFG.SAME_PRODUCT_CLEAN:
                candidate_positions.add(pos)

        clean_positions = sorted(p for p in candidate_positions if 0 <= p <= len(rest))
        clean_positions = rank_candidate_positions(
            batches,
            scenarios,
            rest,
            move_unit,
            product,
            clean_positions,
            max_positions=max(6, max_states),
        )
        block_variants = block_order_variants(batches, scenarios, move_unit)[:3]

        assignment_variants = [
            copy_assignment(plan["assignment"]),
            fastest_assignment_variant_for_block(batches, plan, move_unit),
            load_balanced_assignment_variant_for_block(batches, plan, move_unit),
            campaign_consistent_assignment_variant_for_block(batches, plan, move_unit),
            bottleneck_aware_assignment_variant_for_block(batches, scenarios, plan, move_unit),
        ]

        for p in clean_positions:
            for ordered_block in block_variants:
                new_sequence = rest[:p] + ordered_block + rest[p:]

                for assignment in assignment_variants:
                    candidate = {
                        "sequence": list(new_sequence),
                        "assignment": copy_assignment(assignment),
                    }

                    if not is_valid_plan(batches, candidate):
                        continue

                    signature = (
                        tuple(candidate["sequence"]),
                        tuple(
                            (j, stage, candidate["assignment"][j][stage])
                            for j in range(len(batches))
                            for stage in STAGES
                        ),
                    )

                    if signature in seen_signatures:
                        continue

                    seen_signatures.add(signature)
                    h_energy = mpo_hamiltonian_energy(batches, scenarios, candidate)
                    candidates.append((candidate, h_energy))

    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[:max_states]



def critical_batch_priority(batches, scenarios, plan: Dict[str, Any], batch_id: int) -> float:
    """
    Priority of one batch for QI one-site updates.

    This is not a hidden score.  It uses only shared release, due-date, demand,
    processing-time, and current-position information.  The final decision is
    still made by the common evaluator.
    """
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")
    position = {j: pos for pos, j in enumerate(plan["sequence"])}
    n = len(plan["sequence"])

    due_values = list(avg_due.values())
    demand_values = list(avg_demand.values())
    d_min = min(due_values)
    d_max = max(due_values)
    due_denom = max(1.0, d_max - d_min)
    demand_max = max(1.0, max(demand_values))

    j = batch_id
    fastest_route = sum(min(batches[j]["processing"][stage].values()) for stage in STAGES)
    assigned_route = sum(
        batches[j]["processing"][stage][plan["assignment"][j][stage]]
        for stage in STAGES
    )
    nominal_slack = avg_due[j] - avg_release[j] - fastest_route
    slack_risk = max(0.0, assigned_route - nominal_slack) / max(1.0, fastest_route)

    due_urgency = 1.0 - ((avg_due[j] - d_min) / due_denom)
    demand_weight = avg_demand[j] / demand_max
    position_weight = position[j] / max(1, n - 1)

    product = batches[j]["product"]
    boundary = product_boundary_penalty(batches, plan, product) / max(1.0, CFG.DIFFERENT_GROUP_CLEAN)

    return float(3.0 * slack_risk + 2.0 * due_urgency * demand_weight + position_weight + 0.25 * boundary)


def critical_batches_for_qi(batches, scenarios, plan: Dict[str, Any], max_batches: int) -> List[int]:
    ranked = sorted(
        range(len(batches)),
        key=lambda j: critical_batch_priority(batches, scenarios, plan, j),
        reverse=True,
    )
    return ranked[:max_batches]


def one_batch_candidate_positions(batches, scenarios, rest: List[int], batch_id: int) -> List[int]:
    """Due/release/campaign-aware insertion anchors for a single urgent batch."""
    avg_due = average_scenario_vector(scenarios, "due")
    avg_release = average_scenario_vector(scenarios, "release")
    avg_demand = average_scenario_vector(scenarios, "demand")

    product = batches[batch_id]["product"]
    group = campaign_group(product)
    due = avg_due[batch_id]
    release = avg_release[batch_id]

    positions = {0, len(rest)}
    positions.add(sum(1 for j in rest if avg_due[j] <= due))
    positions.add(sum(1 for j in rest if avg_release[j] <= release))

    # Put the urgent batch near related campaign products when this does not
    # badly violate due-date order.
    for pos, j in enumerate(rest):
        if batches[j]["product"] == product or batches[j]["group"] == group:
            positions.add(pos)
            positions.add(pos + 1)

    # Demand-sensitive anchors: high-demand batches can be useful earlier even
    # when due dates are similar.
    demand_rank_position = sum(
        1
        for j in rest
        if (-avg_demand[j], avg_due[j]) <= (-avg_demand[batch_id], due)
    )
    positions.add(demand_rank_position)

    for anchor in list(positions):
        for delta in [-3, -2, -1, 1, 2, 3]:
            positions.add(anchor + delta)

    return sorted(p for p in positions if 0 <= p <= len(rest))


def local_states_for_critical_batch_site(
    batches,
    scenarios,
    plan: Dict[str, Any],
    batch_id: int,
    max_states: int,
):
    """
    One-site TN local states for an urgent API batch.

    Product-block moves reduce cleaning, but API campaigns also have urgent
    single batches.  This one-site update improves due-date/throughput behavior
    while staying fair: it uses the same public data and consumes the same QI
    evaluation budget.
    """
    sequence = plan["sequence"]
    if batch_id not in sequence:
        return []

    rest = [j for j in sequence if j != batch_id]
    product = batches[batch_id]["product"]
    positions = one_batch_candidate_positions(batches, scenarios, rest, batch_id)
    positions = rank_candidate_positions(
        batches,
        scenarios,
        rest,
        [batch_id],
        product,
        positions,
        max_positions=max(5, max_states),
    )

    assignment_variants = [
        copy_assignment(plan["assignment"]),
        fastest_assignment_variant_for_block(batches, plan, [batch_id]),
        load_balanced_assignment_variant_for_block(batches, plan, [batch_id]),
        bottleneck_aware_assignment_variant_for_block(batches, scenarios, plan, [batch_id]),
    ]

    candidates = []
    seen = set()

    for p in positions:
        new_sequence = rest[:p] + [batch_id] + rest[p:]
        for assignment in assignment_variants:
            candidate = {
                "sequence": list(new_sequence),
                "assignment": copy_assignment(assignment),
            }
            if not is_valid_plan(batches, candidate):
                continue
            sig = plan_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append((candidate, mpo_hamiltonian_energy(batches, scenarios, candidate)))

    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[:max_states]


def high_cleaning_boundaries(batches, plan: Dict[str, Any], max_boundaries: int) -> List[Tuple[int, int, float]]:
    sequence = plan["sequence"]
    out = []
    for idx in range(len(sequence) - 1):
        a = sequence[idx]
        b = sequence[idx + 1]
        clean = setup_time(batches[a]["product"], batches[b]["product"])
        if clean >= CFG.SAME_GROUP_CLEAN:
            out.append((a, b, float(clean)))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:max_boundaries]


def local_states_for_boundary_repair(
    batches,
    scenarios,
    plan: Dict[str, Any],
    left_batch: int,
    right_batch: int,
    max_states: int,
):
    """
    Two-site MPO boundary update for large cleaning/changeover edges.

    It tries small, local repairs around a bad product transition: adjacent swap,
    moving one endpoint near a related campaign block, and faster/bottleneck-aware
    equipment choices for the two endpoint batches.
    """
    sequence = plan["sequence"]
    if left_batch not in sequence or right_batch not in sequence:
        return []

    candidates = []
    seen = set()

    candidate_sequences = []

    # Adjacent two-site swap.
    swapped = list(sequence)
    i = swapped.index(left_batch)
    j = swapped.index(right_batch)
    if abs(i - j) == 1:
        swapped[i], swapped[j] = swapped[j], swapped[i]
        candidate_sequences.append(swapped)

    # Move the right endpoint near product/group relatives.
    for moving in [left_batch, right_batch]:
        rest = [x for x in sequence if x != moving]
        product = batches[moving]["product"]
        group = batches[moving]["group"]
        positions = set(one_batch_candidate_positions(batches, scenarios, rest, moving))
        for pos, x in enumerate(rest):
            if batches[x]["product"] == product or batches[x]["group"] == group:
                positions.add(pos)
                positions.add(pos + 1)
        positions = rank_candidate_positions(
            batches,
            scenarios,
            rest,
            [moving],
            product,
            sorted(p for p in positions if 0 <= p <= len(rest)),
            max_positions=max(4, max_states),
        )
        for p in positions:
            candidate_sequences.append(rest[:p] + [moving] + rest[p:])

    assignment_variants = [
        copy_assignment(plan["assignment"]),
        fastest_assignment_variant_for_block(batches, plan, [left_batch, right_batch]),
        bottleneck_aware_assignment_variant_for_block(batches, scenarios, plan, [left_batch, right_batch]),
    ]

    for seq in candidate_sequences:
        for assignment in assignment_variants:
            candidate = {
                "sequence": list(seq),
                "assignment": copy_assignment(assignment),
            }
            if not is_valid_plan(batches, candidate):
                continue
            sig = plan_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append((candidate, mpo_hamiltonian_energy(batches, scenarios, candidate)))

    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[:max_states]



def qi_mpo_hamiltonian_tn_campaign_repair(
    batches,
    scenarios,
    incumbent_plan: Dict[str, Any],
    time_budget: float,
    trial_id: int,
):
    """
    Stronger fair QI MPO-Hamiltonian tensor-network campaign repair.

    Fairness kept exactly the same:
        - same initial incumbent as SA and CP-SAT hints
        - same final evaluator and same scalar acceptance score for evaluated candidates
        - same release/due/demand scenarios
        - same time budget
        - same candidate-evaluation cap as SA
        - same no-improvement rule as SA
        - same annealing temperature schedule as SA
        - same acceptance rule as SA
        - no precomputed campaign-sorted solution

    The improvement is only representational: QI spends its equal budget on
    API-campaign local states rather than generic random moves.  The search is
    ordered as campaign-group contraction -> product-block contraction ->
    boundary repair -> critical-batch polish.  This prioritizes the main API
    synthesis structure first, then uses small one-site/two-site TN updates for
    due-date and throughput polishing.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    rng = random.Random(45000 + trial_id)

    current = copy_plan(incumbent_plan)
    current_metrics = evaluate_campaign_plan(batches, scenarios, current)
    current_score = current_metrics["Objective"]

    best_plan = copy_plan(current)
    best_metrics = dict(current_metrics)
    best_score = current_score

    evaluated = 1
    skipped_duplicates = 0
    no_improvement_rounds = 0
    sweeps_done = 0

    products = sorted({batch["product"] for batch in batches})
    groups = sorted({batch["group"] for batch in batches})

    def expand_bond_states(bond_states, generator, per_site_cap: int):
        nonlocal evaluated, skipped_duplicates, no_improvement_rounds
        nonlocal best_plan, best_metrics, best_score
        expanded_states = []
        local_improved = False
        seen_this_expansion = set()

        for state_plan, state_score, state_h in bond_states:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break

            expanded_states.append((state_plan, state_score, state_h))
            seen_this_expansion.add(plan_signature(state_plan))

            local_states = generator(state_plan, per_site_cap)

            for candidate_plan, candidate_h in local_states:
                if (
                    time.perf_counter() >= deadline
                    or evaluated >= CFG.LOCAL_MAX_EVALS
                    or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
                ):
                    break

                sig = plan_signature(candidate_plan)
                if sig in seen_this_expansion:
                    skipped_duplicates += 1
                    continue
                seen_this_expansion.add(sig)

                candidate_metrics = evaluate_campaign_plan(batches, scenarios, candidate_plan)
                candidate_score = candidate_metrics["Objective"]
                evaluated += 1

                temperature = annealing_temperature(evaluated)

                accepted = accept_move(candidate_score, state_score, temperature, rng)
                if accepted:
                    expanded_states.append((copy_plan(candidate_plan), candidate_score, candidate_h))

                if candidate_score < best_score:
                    best_plan = copy_plan(candidate_plan)
                    best_score = candidate_score
                    best_metrics = dict(candidate_metrics)
                    no_improvement_rounds = 0
                    local_improved = True
                else:
                    no_improvement_rounds += 1

        if not expanded_states:
            return bond_states, local_improved

        expanded_states = sorted(expanded_states, key=lambda x: (x[1], x[2]))
        return expanded_states[:CFG.MPO_BOND_DIM], local_improved

    while (
        time.perf_counter() < deadline
        and evaluated < CFG.LOCAL_MAX_EVALS
        and no_improvement_rounds < CFG.MAX_NO_IMPROVEMENT_ROUNDS
        and sweeps_done < CFG.MPO_SWEEPS
    ):
        sweeps_done += 1

        bond_states = [
            (
                copy_plan(current),
                current_score,
                mpo_hamiltonian_energy(batches, scenarios, current),
            )
        ]

        # 1. Campaign-group contractions first.  In API synthesis, this is the
        # strongest structural move for cleaning/changeover and makespan.
        sweep_groups = sorted(
            groups,
            key=lambda g: group_urgency_score(batches, scenarios, current, g),
            reverse=True,
        )
        for group in sweep_groups:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break
            bond_states, improved = expand_bond_states(
                bond_states,
                lambda state_plan, cap, g=group: local_states_for_campaign_group_block(
                    batches, scenarios, state_plan, g, cap
                ),
                per_site_cap=max(8, CFG.MPO_LOCAL_STATES // 2),
            )
            _ = improved

        # 2. Product-level TN contraction with internal due/demand ordering.
        sweep_products = sorted(
            products,
            key=lambda p: product_urgency_score(batches, scenarios, current, p),
            reverse=True,
        )
        for product in sweep_products:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break
            bond_states, improved = expand_bond_states(
                bond_states,
                lambda state_plan, cap, p=product: local_states_for_api_product_block(
                    batches, scenarios, state_plan, p, cap
                ),
                per_site_cap=CFG.MPO_LOCAL_STATES,
            )
            _ = improved

        # 3. A second group pass, because product moves can open new useful
        # campaign-group contractions.  This still uses the same QI cap/time.
        for group in sweep_groups:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break
            bond_states, improved = expand_bond_states(
                bond_states,
                lambda state_plan, cap, g=group: local_states_for_campaign_group_block(
                    batches, scenarios, state_plan, g, cap
                ),
                per_site_cap=max(6, CFG.MPO_LOCAL_STATES // 2),
            )
            _ = improved

        # 4. Two-site boundary repair for the largest remaining cleaning edges.
        reference_plan = bond_states[0][0] if bond_states else current
        boundaries = high_cleaning_boundaries(
            batches,
            reference_plan,
            max_boundaries=min(4, len(batches) - 1),
        )
        for left_batch, right_batch, _clean in boundaries:
            if (
                time.perf_counter() >= deadline
                or evaluated >= CFG.LOCAL_MAX_EVALS
                or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
            ):
                break
            bond_states, improved = expand_bond_states(
                bond_states,
                lambda state_plan, cap, a=left_batch, b=right_batch: local_states_for_boundary_repair(
                    batches, scenarios, state_plan, a, b, cap
                ),
                per_site_cap=max(4, CFG.MPO_LOCAL_STATES // 3),
            )
            _ = improved

        # 5. Critical-batch polish only after the campaign structure has been
        # repaired.  This avoids sacrificing cleaning/makespan just to chase a
        # few single-batch due-date moves.
        if evaluated < int(0.90 * CFG.LOCAL_MAX_EVALS):
            critical_batches = critical_batches_for_qi(
                batches,
                scenarios,
                bond_states[0][0] if bond_states else current,
                max_batches=min(4, len(batches)),
            )
            for batch_id in critical_batches:
                if (
                    time.perf_counter() >= deadline
                    or evaluated >= CFG.LOCAL_MAX_EVALS
                    or no_improvement_rounds >= CFG.MAX_NO_IMPROVEMENT_ROUNDS
                ):
                    break
                bond_states, improved = expand_bond_states(
                    bond_states,
                    lambda state_plan, cap, b=batch_id: local_states_for_critical_batch_site(
                        batches, scenarios, state_plan, b, cap
                    ),
                    per_site_cap=max(3, CFG.MPO_LOCAL_STATES // 4),
                )
                _ = improved

        if bond_states:
            current, current_score, _ = bond_states[0]

        # The no-improvement counter is updated per evaluated candidate inside
        # expand_bond_states, matching the SA stopping rule. It is not updated
        # once per sweep, because that would give QI a weaker stopping rule.

    runtime = time.perf_counter() - t0

    info = (
        f"Status=OK; "
        f"method=QI_MPO_Hamiltonian_TN_API_Campaign_StrongerFairFourMetric; "
        f"evaluated_candidates={evaluated}; "
        f"skipped_duplicate_states={skipped_duplicates}; "
        f"sweeps_done={sweeps_done}; "
        f"bond_dim={CFG.MPO_BOND_DIM}; "
        f"local_states={CFG.MPO_LOCAL_STATES}; "
        f"start=shared_incumbent; "
        f"time_budget={time_budget}; "
        f"candidate_cap={CFG.LOCAL_MAX_EVALS}; "
        f"no_improvement_cap={CFG.MAX_NO_IMPROVEMENT_ROUNDS}; "
        f"final_no_improvement_count={no_improvement_rounds}; "
        f"same_acceptance_rule_as_SA; "
        f"local_state=campaign_group+product_block+boundary_repair+critical_batch+equipment_assignment"
    )

    return best_plan, best_metrics, runtime, info


# ============================================================
# 8. CP-SAT FLEXIBLE EQUIPMENT MODEL
# ============================================================


def nominal_scenario_from_scenarios(scenarios) -> Dict[str, List[int]]:
    n = len(scenarios[0]["release"])
    nominal = {"release": [], "due": [], "demand": []}

    for j in range(n):
        nominal["release"].append(int(round(np.mean([s["release"][j] for s in scenarios]))))
        nominal["due"].append(int(round(np.mean([s["due"][j] for s in scenarios]))))
        nominal["demand"].append(int(round(np.mean([s["demand"][j] for s in scenarios]))))

    return nominal


def horizon_bound(batches, scenarios) -> int:
    max_route = 0
    for batch in batches:
        for stage in STAGES:
            max_route += max(batch["processing"][stage].values())

    release_bound = max(max(scenario["release"]) for scenario in scenarios)
    setup_bound = CFG.DIFFERENT_GROUP_CLEAN * len(batches) * len(STAGES)

    return int(max_route + release_bound + setup_bound + 2000)


def return_incumbent_result(
    batches,
    scenarios,
    incumbent_plan: Dict[str, Any],
    runtime: float,
    reason: str,
):
    metrics = evaluate_campaign_plan(batches, scenarios, incumbent_plan)

    return (
        copy_plan(incumbent_plan),
        metrics,
        runtime,
        f"Status=RETURNED_INCUMBENT; reason={reason}",
    )


def cpsat_flexible_equipment_model(
    batches,
    scenarios,
    incumbent_plan: Dict[str, Any],
    time_budget: float,
):
    """
    CP-SAT flexible equipment model for API synthesis campaign scheduling.

    The CP-SAT model chooses equipment assignments and a nominal schedule for
    the average scenario. The resulting equipment assignment and batch priority
    sequence are then evaluated under all scenarios by the same final evaluator
    used for every method.

    Fairness:
        - same batches
        - same four-stage route
        - same compatible equipment sets
        - same processing times
        - same cleaning/changeover matrix
        - same time budget
        - same incumbent used as a warm-start hint
        - same final scenario evaluator
    """
    t0 = time.perf_counter()

    n = len(batches)
    nominal = nominal_scenario_from_scenarios(scenarios)
    H = horizon_bound(batches, scenarios)

    model = cp_model.CpModel()

    start = {}
    end = {}
    x = {}

    for j in range(n):
        for stage in STAGES:
            start[j, stage] = model.NewIntVar(0, H, f"s_{j}_{stage}")
            end[j, stage] = model.NewIntVar(0, H, f"e_{j}_{stage}")

            options = compatible_equipment(batches[j]["product"], stage)
            option_bools = []

            for eq in options:
                x[j, stage, eq] = model.NewBoolVar(f"x_{j}_{stage}_{eq}")
                option_bools.append(x[j, stage, eq])

            model.AddExactlyOne(option_bools)

            # Processing time is selected by equipment assignment.
            model.Add(
                end[j, stage]
                == start[j, stage]
                + sum(
                    batches[j]["processing"][stage][eq] * x[j, stage, eq]
                    for eq in options
                )
            )

    # Raw-material release before synthesis.
    for j in range(n):
        model.Add(start[j, "SYNTHESIS"] >= nominal["release"][j])

    # Stage precedence.
    for j in range(n):
        for stage_a, stage_b in zip(STAGES[:-1], STAGES[1:]):
            model.Add(start[j, stage_b] >= end[j, stage_a])

    bool_count = 0

    # Sequence-dependent setup constraints on each equipment unit.
    for eq, eq_stage in STAGE_OF_EQUIPMENT.items():
        compatible_ops = [
            j for j in range(n)
            if eq in compatible_equipment(batches[j]["product"], eq_stage)
        ]

        for idx_a in range(len(compatible_ops)):
            for idx_b in range(idx_a + 1, len(compatible_ops)):
                if time.perf_counter() - t0 > time_budget:
                    return return_incumbent_result(
                        batches,
                        scenarios,
                        incumbent_plan,
                        time.perf_counter() - t0,
                        f"model_build_used_budget after {bool_count} order binaries",
                    )

                i = compatible_ops[idx_a]
                j = compatible_ops[idx_b]
                y = model.NewBoolVar(f"y_{i}_before_{j}_on_{eq}")
                bool_count += 1

                setup_ij = setup_time(batches[i]["product"], batches[j]["product"])
                setup_ji = setup_time(batches[j]["product"], batches[i]["product"])

                model.Add(
                    start[j, eq_stage] >= end[i, eq_stage] + setup_ij
                ).OnlyEnforceIf([x[i, eq_stage, eq], x[j, eq_stage, eq], y])

                model.Add(
                    start[i, eq_stage] >= end[j, eq_stage] + setup_ji
                ).OnlyEnforceIf([x[i, eq_stage, eq], x[j, eq_stage, eq], y.Not()])

    Cmax = model.NewIntVar(0, H, "Cmax")
    for j in range(n):
        model.Add(Cmax >= end[j, "PACKAGING"])

    tardiness_vars = []
    for j in range(n):
        tardy = model.NewIntVar(0, H, f"tardy_{j}")
        model.Add(tardy >= end[j, "PACKAGING"] - nominal["due"][j])
        model.Add(tardy >= 0)
        tardiness_vars.append(tardy)

    model.Minimize(
        Cmax
        + CFG.CPSAT_TARDINESS_WEIGHT * sum(tardiness_vars)
    )

    # Warm-start hints from the shared incumbent decoded on the nominal scenario.
    inc_metrics = decode_one_scenario(batches, nominal, incumbent_plan)
    _ = inc_metrics  # kept for transparency/debugging if printed later

    # Reconstruct an approximate incumbent schedule for hints.
    # The hints are optional; final fairness comes from the common evaluator.
    for j in range(n):
        for stage in STAGES:
            eq_hint = incumbent_plan["assignment"][j][stage]
            for eq in compatible_equipment(batches[j]["product"], stage):
                model.AddHint(x[j, stage, eq], 1 if eq == eq_hint else 0)

    build_time = time.perf_counter() - t0
    remaining = time_budget - build_time

    if remaining <= 0.01:
        return return_incumbent_result(
            batches,
            scenarios,
            incumbent_plan,
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
            batches,
            scenarios,
            incumbent_plan,
            runtime,
            "no_solution_within_budget",
        )

    assignment = {}
    for j in range(n):
        assignment[j] = {}
        for stage in STAGES:
            chosen_eq = None
            for eq in compatible_equipment(batches[j]["product"], stage):
                if solver.Value(x[j, stage, eq]) == 1:
                    chosen_eq = eq
                    break
            if chosen_eq is None:
                chosen_eq = incumbent_plan["assignment"][j][stage]
            assignment[j][stage] = chosen_eq

    # Convert CP-SAT schedule into the shared plan representation.
    # Batch priority is based on nominal synthesis start time.
    sequence = sorted(
        range(n),
        key=lambda j: (solver.Value(start[j, "SYNTHESIS"]), j),
    )

    plan = {
        "sequence": sequence,
        "assignment": assignment,
    }

    assert is_valid_plan(batches, plan)

    metrics = evaluate_campaign_plan(batches, scenarios, plan)
    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    info = (
        f"Status={status_name}; "
        f"order_binaries={bool_count}; "
        f"start=shared_incumbent_hint; "
        f"time_budget={time_budget}; "
        f"nominal_model=evaluated_robustly_after_solve"
    )

    return plan, metrics, runtime, info


# ============================================================
# 9. BENCHMARK DRIVER
# ============================================================


def method_row(trial: int, method: str, metrics: Dict[str, float], runtime: float, info: str):
    return {
        "Trial": int(trial),
        "Method": str(method),
        # Four reported business metrics.
        "ThroughputKgPerTime": float(metrics["ThroughputKgPerTime"]),
        "Makespan": float(metrics["Makespan"]),
        "CleaningChangeoverTime": float(metrics["CleaningChangeoverTime"]),
        "DueDateHitRate": float(metrics["DueDateHitRate"]),
        # Audit-only fields, not used in the comparison summary.
        "Objective_AuditOnly": float(metrics["Objective"]),
        "WeightedTardiness_AuditOnly": float(metrics["WeightedTardiness"]),
        "OnTimeThroughputKg_AuditOnly": float(metrics["OnTimeThroughputKg"]),
        "OnTimeThroughputRate_AuditOnly": float(metrics["OnTimeThroughputRate"]),
        "Runtime_s_AuditOnly": float(runtime),
        "Info": str(info),
    }


def run_trial(trial_id: int):
    batches, scenarios = generate_trial_case(trial_id)

    # Shared incumbent is generated exactly once.
    t0 = time.perf_counter()
    incumbent_plan = incumbent_mixed_campaign_plan(batches, trial_id)
    validate_trial_fairness(batches, scenarios, incumbent_plan)

    incumbent_metrics = evaluate_campaign_plan(batches, scenarios, incumbent_plan)
    incumbent_runtime = time.perf_counter() - t0

    rows = []

    rows.append(
        method_row(
            trial_id,
            "Incumbent Mixed Campaign",
            incumbent_metrics,
            incumbent_runtime,
            "Status=OK; shared starting plan; mixed product campaign",
        )
    )

    cp_plan, cp_metrics, cp_runtime, cp_info = cpsat_flexible_equipment_model(
        batches,
        scenarios,
        incumbent_plan,
        CFG.TIME_BUDGET_SECONDS,
    )

    rows.append(
        method_row(
            trial_id,
            "CP-SAT Flexible Equipment Model",
            cp_metrics,
            cp_runtime,
            cp_info,
        )
    )

    sa_plan, sa_metrics, sa_runtime, sa_info = classical_structured_simulated_annealing(
        batches,
        scenarios,
        incumbent_plan,
        CFG.TIME_BUDGET_SECONDS,
        trial_id,
    )

    rows.append(
        method_row(
            trial_id,
            "Classical Structured Simulated Annealing",
            sa_metrics,
            sa_runtime,
            sa_info,
        )
    )

    qi_plan, qi_metrics, qi_runtime, qi_info = qi_mpo_hamiltonian_tn_campaign_repair(
        batches,
        scenarios,
        incumbent_plan,
        CFG.TIME_BUDGET_SECONDS,
        trial_id,
    )

    rows.append(
        method_row(
            trial_id,
            "QI MPO-Hamiltonian TN Campaign Repair",
            qi_metrics,
            qi_runtime,
            qi_info,
        )
    )

    assert is_valid_plan(batches, cp_plan)
    assert is_valid_plan(batches, sa_plan)
    assert is_valid_plan(batches, qi_plan)

    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize only the four requested business metrics."""
    summary = (
        results.groupby("Method", dropna=False)
        .agg(
            Mean_ThroughputKgPerTime=("ThroughputKgPerTime", "mean"),
            Mean_Makespan=("Makespan", "mean"),
            Mean_CleaningChangeoverTime=("CleaningChangeoverTime", "mean"),
            Mean_DueDateHitRate=("DueDateHitRate", "mean"),
        )
        .reset_index()
    )

    metric_cols = [c for c in summary.columns if c != "Method"]
    summary[metric_cols] = (
        summary[metric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .round(6)
    )

    summary["Throughput_Rank"] = summary["Mean_ThroughputKgPerTime"].rank(method="min", ascending=False).astype(int)
    summary["Makespan_Rank"] = summary["Mean_Makespan"].rank(method="min", ascending=True).astype(int)
    summary["Cleaning_Rank"] = summary["Mean_CleaningChangeoverTime"].rank(method="min", ascending=True).astype(int)
    summary["DueDate_Rank"] = summary["Mean_DueDateHitRate"].rank(method="min", ascending=False).astype(int)

    rank_cols = ["Throughput_Rank", "Makespan_Rank", "Cleaning_Rank", "DueDate_Rank"]
    summary["Combined_4Metric_RankScore"] = summary[rank_cols].sum(axis=1).astype(int)
    summary = summary.sort_values(["Combined_4Metric_RankScore", "Makespan_Rank", "Cleaning_Rank"]).reset_index(drop=True)

    return summary

def plot_bar(summary: pd.DataFrame, metric: str, title: str, ylabel: str, higher_is_better: bool = False, logy: bool = False):
    data = summary.copy()
    data = data.sort_values(metric, ascending=not higher_is_better)

    plt.figure(figsize=(12, 4.8))

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
                label = f"{value:.4f}"
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
    print("All methods are evaluated on the same API synthesis campaign instances.")
    print("For each trial, the API batches, products, processing times, compatible")
    print("equipment sets, release-time scenarios, due-date scenarios, demand forecasts,")
    print("cleaning/changeover matrix, initial incumbent plan, decoder, final")
    print("evaluation function, and four reported metrics are identical across methods.")
    print("Classical structured SA and QI use the same scalar acceptance score for evaluated candidates;")
    print("CP-SAT optimizes a nominal finite-horizon model and is then evaluated")
    print("by the same final multi-scenario evaluator as every other method.")
    print("CP-SAT, classical simulated annealing, and QI receive the same maximum")
    print("wall-clock time budget. CP-SAT uses one worker by default so it does")
    print("not receive extra CPU parallelism relative to the single-threaded heuristics.")
    print("Classical structured simulated annealing and QI receive the same candidate-evaluation cap,")
    print("same no-improvement stopping rule, same annealing temperature schedule,")
    print("and same acceptance rule. SA is also allowed to sample from the same")
    print("campaign-aware block/equipment neighborhoods that QI uses, so QI does")
    print("not have exclusive access to stronger local moves.")
    print("QI is not given a precomputed campaign-sorted sequence.")
    print("QI's possible advantage comes only from its MPO/Hamiltonian tensor-network")
    print("representation: API products are sites, block positions/equipment choices")
    print("are local states, and the bond dimension controls retained candidate states.")


def print_problem_mapping():
    print("\nAPI SYNTHESIS CAMPAIGN PROBLEM MAPPING")
    print("--------------------------------------")
    print("Which API product should be produced first?")
    print("    Encoded by the campaign batch priority sequence.")
    print("Which reactor/equipment should be assigned to each batch?")
    print("    Encoded by stage-wise equipment assignment for synthesis, purification,")
    print("    drying, and packaging.")
    print("How to minimize changeovers?")
    print("    Measured by CleaningChangeoverTime.")
    print("How to meet delivery deadlines and demand forecasts?")
    print("    Measured by DueDateHitRate and ThroughputKgPerTime.")
    print("How to coordinate synthesis, purification, drying, and packaging?")
    print("    Enforced by the common campaign decoder with stage precedence constraints.")


def print_qi_interpretation():
    print("\nMPO-HAMILTONIAN QI INTERPRETATION")
    print("---------------------------------")
    print("The QI method is tensor-network-inspired rather than quantum-hardware-based.")
    print("The schedule is represented through API product campaign blocks plus")
    print("equipment-assignment states.")
    print("Each API product acts like a tensor-network site.")
    print("Each local state is a block insertion position together with feasible")
    print("reactor/purification/dryer/packaging assignments.")
    print("The Hamiltonian surrogate has cleaning, due-date, demand-throughput,")
    print("equipment-load, and fragmentation terms.")
    print("The cleaning term is a nearest-neighbor MPO-like interaction.")
    print("The due-date and demand terms are one-site fields.")
    print("The bond dimension controls how many candidate campaign states are retained.")
    print("Final reported scores are not Hamiltonian scores; they are evaluated by the")
    print("same campaign decoder and metric functions used for every method.")


def print_metric_definitions():
    print("\nMETRIC DEFINITIONS")
    print("------------------")
    print("Mean_ThroughputKgPerTime:")
    print("    Average produced demand divided by campaign completion time. Higher is better.")
    print("Mean_Makespan:")
    print("    Mean final completion time over all release/due/demand scenarios and trials. Lower is better.")
    print("Mean_CleaningChangeoverTime:")
    print("    Mean sequence-dependent setup/cleaning/changeover time across equipment. Lower is better.")
    print("Mean_DueDateHitRate:")
    print("    Mean fraction of batches completed by their due dates. Higher is better.")
    print("Runtime and the scalar internal objective are saved only as audit fields in the run-level CSV; they are not compared metrics.")

def main():
    random.seed(CFG.GLOBAL_SEED)
    np.random.seed(CFG.GLOBAL_SEED)

    print("Fair API Synthesis Campaign Scheduling Benchmark with MPO-Hamiltonian QI Method")
    print("=============================================================================")
    print(f"Trials: {CFG.N_TRIALS}")
    print(f"API batches per trial: {CFG.N_API_BATCHES}")
    print(f"API products: {CFG.N_API_PRODUCTS}")
    print(f"Release/due/demand scenarios per trial: {CFG.N_SCENARIOS}")
    print(f"Stages: {' -> '.join(STAGES)}")
    print(f"Equal maximum time budget for CP-SAT, SA, and QI: {CFG.TIME_BUDGET_SECONDS:.3f}s")
    print(f"Equal candidate cap for SA and QI: {CFG.LOCAL_MAX_EVALS}")
    print(f"Equal no-improvement cap for SA and QI: {CFG.MAX_NO_IMPROVEMENT_ROUNDS}")
    print(f"MPO bond dimension: {CFG.MPO_BOND_DIM}")
    print(f"MPO local states: {CFG.MPO_LOCAL_STATES}")
    print(f"MPO sweeps: {CFG.MPO_SWEEPS}")
    print(f"CP-SAT workers: {CFG.NUM_WORKERS}  (single-worker default for CPU fairness)")
    print("\nGurobi is intentionally removed because size-limited licenses can distort results.")

    print_problem_mapping()

    print("\nOnly these four business metrics are compared:")
    print("1. Mean_ThroughputKgPerTime  higher is better")
    print("2. Mean_Makespan             lower is better")
    print("3. Mean_CleaningChangeoverTime lower is better")
    print("4. Mean_DueDateHitRate       higher is better")

    all_rows = []

    for trial in range(CFG.N_TRIALS):
        rows = run_trial(trial)
        all_rows.extend(rows)

        if (trial + 1) % 5 == 0 or (trial + 1) == CFG.N_TRIALS:
            partial = pd.DataFrame(all_rows)
            partial_summary = summarize(partial)

            print(f"\nCompleted {trial + 1}/{CFG.N_TRIALS} trials")
            print(partial_summary[
                [
                    "Method",
                    "Mean_ThroughputKgPerTime",
                    "Mean_Makespan",
                    "Mean_CleaningChangeoverTime",
                    "Mean_DueDateHitRate",
                    "Combined_4Metric_RankScore",
                ]
            ])

    results = pd.DataFrame(all_rows)

    numeric_cols = [
        "ThroughputKgPerTime",
        "Makespan",
        "CleaningChangeoverTime",
        "DueDateHitRate",
        "Objective_AuditOnly",
        "WeightedTardiness_AuditOnly",
        "OnTimeThroughputKg_AuditOnly",
        "OnTimeThroughputRate_AuditOnly",
        "Runtime_s_AuditOnly",
    ]

    for col in numeric_cols:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    summary = summarize(results)

    print("\nRUN-LEVEL RESULTS")
    print(results)

    print("\nSUMMARY")
    print(summary)

    results.to_csv("fair_api_synthesis_four_metrics_run_level_audit.csv", index=False)
    summary.to_csv("fair_api_synthesis_four_metrics_summary.csv", index=False)

    print("\nSaved:")
    print("fair_api_synthesis_four_metrics_run_level_audit.csv")
    print("fair_api_synthesis_four_metrics_summary.csv")

    plot_bar(
        summary,
        "Mean_ThroughputKgPerTime",
        "Mean Throughput",
        "Mean throughput kg per time, higher is better",
        higher_is_better=True,
    )

    plot_bar(
        summary,
        "Mean_Makespan",
        "Mean Makespan",
        "Mean makespan, lower is better",
    )

    plot_bar(
        summary,
        "Mean_CleaningChangeoverTime",
        "Mean Cleaning/Changeover Time",
        "Mean cleaning/changeover time, lower is better",
    )

    plot_bar(
        summary,
        "Mean_DueDateHitRate",
        "Mean Due-Date Hit Rate",
        "Mean due-date hit rate, higher is better",
        higher_is_better=True,
    )

    print_fairness_statement()
    print_qi_interpretation()
    print_metric_definitions()

    print("\nINTERPRETATION NOTE")
    print("-------------------")
    print("Only four metrics are compared: throughput, makespan, cleaning/changeover,")
    print("and due-date hit rate.")
    print("QI should be interpreted as having a structural representation advantage only")
    print("if it improves these metrics under the same starting plan, same time budget,")
    print("same candidate cap, same decoder, and same final evaluator.")


if __name__ == "__main__":
    main()
