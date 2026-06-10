import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from jsp_qubo_benchmark import (
    BENCHMARK_LABELS,
    benchmark_fixed_C,
    compute_reduction_ratios,
    normalize_benchmarks,
    save_comparison_plot,
    solve_jsp_cpsat_optimize,
    solve_jsp_tn_lns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orlib_jobshop1_jssp_instances_with_bounds.json"
)


def timestamp_string() -> str:
    """
    Return a compact timestamp like:
        20260608223315
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def load_dataset(path: Path = DATASET_PATH) -> dict:
    """
    Load the parsed OR-Library + bounds JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file:\n"
            f"  {path}\n\n"
            f"Expected location:\n"
            f"  data/processed/orlib_jobshop1_jssp_instances_with_bounds.json"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_jobs_data(instance_record: dict):
    """
    Convert JSON jobs_data into the tuple-pair format used by jsp_qubo_benchmark.py.

    JSON stores:
        jobs_data[j][k] = [machine, duration]

    The core benchmark code works with:
        jobs_data[j][k] = (machine, duration)
    """
    return [
        [(int(machine), int(duration)) for machine, duration in job]
        for job in instance_record["jobs_data"]
    ]


def choose_c_values(instance_record: dict, opt_makespan: int | None = None) -> list[int]:
    """
    Choose fixed makespan values C.

    Priority:
    1. If exact optimum is known, test around it.
    2. If only lower/upper bounds are known, test lower bound, midpoint, upper bound.
    3. If CP-SAT found a makespan, test around that found makespan.
    4. Otherwise fall back to a crude horizon.
    """
    optimum = instance_record.get("optimum")
    lower_bound = instance_record.get("lower_bound")
    upper_bound = instance_record.get("upper_bound")

    if optimum is not None:
        C = int(optimum)
        return sorted(
            {
                max(1, C - 2),
                max(1, C - 1),
                C,
                C + 1,
                C + 3,
                C + 5,
            }
        )

    if lower_bound is not None and upper_bound is not None:
        lb = int(lower_bound)
        ub = int(upper_bound)
        midpoint = (lb + ub) // 2
        return sorted({lb, midpoint, ub})

    if opt_makespan is not None:
        C = int(opt_makespan)
        return sorted({max(1, C - 1), C, C + 1, C + 3})

    horizon = int(instance_record["horizon_total_processing_time"])
    return [max(1, horizon // 2), horizon]


def make_run_output_dir(base_output_dir: Path, instance_name: str, run_timestamp: str) -> Path:
    """
    Create a unique output directory for one instance.

    Example:
        outputs/abz5-20260608223315
    """
    candidate = base_output_dir / f"{instance_name}-{run_timestamp}"

    # Rare case: same timestamp folder already exists.
    counter = 2
    while candidate.exists():
        candidate = base_output_dir / f"{instance_name}-{run_timestamp}-{counter}"
        counter += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def print_instance_summary(instance_name: str, instance_record: dict):
    """
    Print useful metadata for one selected instance.
    """
    print("=" * 70)
    print(f"JSP-QUBO Benchmark: {instance_name}")
    print("=" * 70)
    print(f"Description: {instance_record.get('description', '')}")
    print(f"Family: {instance_record.get('family', '')}")
    print(f"Jobs: {instance_record['n_jobs']}")
    print(f"Machines: {instance_record['n_machines']}")
    print(f"Operations: {instance_record['n_operations']}")
    print(f"Total processing time: {instance_record['total_processing_time']}")
    print(f"Max job load: {instance_record['max_job_load']}")
    print(f"Max machine load: {instance_record['max_machine_load']}")
    print()

    print("Known makespan metadata:")
    print(f"  lower_bound: {instance_record.get('lower_bound')}")
    print(f"  upper_bound: {instance_record.get('upper_bound')}")
    print(f"  optimum: {instance_record.get('optimum')}")
    print(f"  status: {instance_record.get('status')}")
    print(f"  source: {instance_record.get('bounds_source')}")
    print()


def save_run_metadata(
    output_dir: Path,
    instance_name: str,
    instance_record: dict,
    c_values: list[int],
    args,
    run_timestamp: str,
    cpsat_opt_ref: dict | None,
):
    """
    Save a small JSON file describing one instance benchmark run.
    """
    metadata = {
        "instance": instance_name,
        "timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "c_values": c_values,
        "run_settings": {
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "seed": args.seed,
            "cpsat_time_limit": args.cpsat_time_limit,
            "max_workers_total": args.max_workers,
            "skip_cpsat_optimize": args.skip_cpsat_optimize,
            "benchmarks": list(args.benchmarks),
        },
        "cpsat_optimize_reference": cpsat_opt_ref,
        "instance_metadata": {
            "description": instance_record.get("description"),
            "family": instance_record.get("family"),
            "n_jobs": instance_record.get("n_jobs"),
            "n_machines": instance_record.get("n_machines"),
            "n_operations": instance_record.get("n_operations"),
            "total_processing_time": instance_record.get("total_processing_time"),
            "horizon_total_processing_time": instance_record.get(
                "horizon_total_processing_time"
            ),
            "max_job_load": instance_record.get("max_job_load"),
            "max_machine_load": instance_record.get("max_machine_load"),
            "lower_bound": instance_record.get("lower_bound"),
            "upper_bound": instance_record.get("upper_bound"),
            "optimum": instance_record.get("optimum"),
            "status": instance_record.get("status"),
            "bounds_source": instance_record.get("bounds_source"),
            "bounds_source_url": instance_record.get("bounds_source_url"),
            "instance_data_source": instance_record.get("instance_data_source"),
            "instance_data_source_url": instance_record.get(
                "instance_data_source_url"
            ),
        },
    }

    metadata_path = output_dir / f"{instance_name}_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_outputs(
    results_df: pd.DataFrame,
    c_values: list[int],
    output_dir: Path,
    instance_name: str,
):
    """
    Save CSV outputs and comparison plots for one instance.
    """
    results_path = output_dir / f"{instance_name}_jsp_qubo_benchmark_results.csv"
    ratios_path = output_dir / f"{instance_name}_jsp_qubo_reduction_ratios.csv"

    results_df.to_csv(results_path, index=False)

    ratios_df = compute_reduction_ratios(results_df, c_values)
    ratios_df.insert(0, "instance", instance_name)
    ratios_df.to_csv(ratios_path, index=False)

    qubo_results = results_df[
        results_df["model"].isin(
            [
                BENCHMARK_LABELS["time_indexed_qubo"],
                BENCHMARK_LABELS["compact_disjunctive_qubo"],
            ]
        )
    ].copy()

    save_comparison_plot(
        qubo_results,
        "binary_vars",
        "Binary variables",
        f"{instance_name}: QUBO variable count comparison",
        output_dir / f"{instance_name}_qubo_variable_count_comparison.png",
    )

    save_comparison_plot(
        qubo_results,
        "quadratic_terms",
        "Quadratic terms",
        f"{instance_name}: QUBO quadratic terms comparison",
        output_dir / f"{instance_name}_qubo_quadratic_terms_comparison.png",
    )

    save_comparison_plot(
        results_df,
        "time_sec",
        "Runtime (s)",
        f"{instance_name}: benchmark runtime comparison",
        output_dir / f"{instance_name}_benchmark_runtime_comparison.png",
    )

    save_comparison_plot(
        results_df,
        "makespan",
        "Makespan",
        f"{instance_name}: benchmark makespan comparison",
        output_dir / f"{instance_name}_benchmark_makespan_comparison.png",
    )

    feasibility_results = results_df.copy()
    feasibility_results["feasible_found"] = (
        feasibility_results["feasible_found"].fillna(False).astype(bool).astype(int)
    )

    save_comparison_plot(
        feasibility_results,
        "feasible_found",
        "Feasible found (1=yes, 0=no)",
        f"{instance_name}: benchmark feasibility comparison",
        output_dir / f"{instance_name}_benchmark_feasibility_comparison.png",
    )

    print()
    print(f"Saved outputs for {instance_name}:")
    print(f"  {results_path}")
    print(f"  {ratios_path}")
    print(f"  {output_dir / f'{instance_name}_qubo_variable_count_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_qubo_quadratic_terms_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_benchmark_runtime_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_benchmark_makespan_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_benchmark_feasibility_comparison.png'}")


def run_one_instance_c_task(task: dict):
    """
    Worker function for one independent benchmark task:
        one instance, one fixed makespan C.

    This function must be top-level for Windows multiprocessing.
    """
    df = benchmark_fixed_C(
        task["jobs_data"],
        task["C"],
        num_reads=task["num_reads"],
        num_sweeps=task["num_sweeps"],
        seed=task["seed"],
        cpsat_time_limit=task["cpsat_time_limit"],
        benchmarks=task["benchmarks"],
        known_optimum=task["known_optimum"],
        tn_result=task.get("tn_result"),
    )

    return task["instance_name"], task["C"], df


def add_result_metadata(
    df: pd.DataFrame,
    instance_name: str,
    instance_record: dict,
    output_dir: Path,
    run_timestamp: str,
) -> pd.DataFrame:
    """
    Add instance/run metadata columns to the result DataFrame.
    """
    df = df.copy()

    df.insert(0, "instance", instance_name)
    df.insert(1, "known_lower_bound", instance_record.get("lower_bound"))
    df.insert(2, "known_upper_bound", instance_record.get("upper_bound"))
    df.insert(3, "known_optimum", instance_record.get("optimum"))
    df.insert(4, "known_status", instance_record.get("status"))
    df.insert(5, "run_timestamp", run_timestamp)
    df.insert(6, "run_output_dir", str(output_dir))

    return df


def parse_instance_names_from_input(raw: str) -> list[str]:
    """
    Accept either spaces or commas:
        ft06 la01
        ft06, la01
    """
    cleaned = raw.replace(",", " ")
    return [x.strip().lower() for x in cleaned.split() if x.strip()]


def select_instance_names(args, instances: dict) -> list[str]:
    """
    Select one or more instance names from CLI args or interactive prompt.
    """
    if args.instances is None:
        print("Available instance examples:")
        print("  ft06, ft10, ft20")
        print("  la01, la02, ..., la40")
        print("  abz5, abz6, ..., abz9")
        print("  orb01, ..., orb10")
        print("  swv01, ..., swv20")
        print("  yn1, ..., yn4")
        print()
        raw = input(
            "Which instance(s) do you want to run? "
            "Use spaces or commas, e.g. ft06 la01: "
        )
        instance_names = parse_instance_names_from_input(raw)
    else:
        instance_names = [name.strip().lower() for name in args.instances]

    if not instance_names:
        raise ValueError("No instances were selected.")

    unknown = [name for name in instance_names if name not in instances]
    if unknown:
        names_preview = ", ".join(sorted(instances)[:30])
        raise ValueError(
            f"Unknown instance(s): {', '.join(unknown)}\n"
            f"First available names include:\n"
            f"  {names_preview}, ..."
        )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(instance_names))


def normalize_custom_c_values(c_values: list[int] | None) -> list[int] | None:
    """
    Remove duplicate custom C-values while preserving order.
    """
    if c_values is None:
        return None

    return list(dict.fromkeys(int(C) for C in c_values))


def run_instance_benchmark(args):
    """
    Main wrapper logic.

    Supports:
    - one instance or many instances;
    - one C-value or many C-values;
    - parallel execution over all (instance, C) pairs;
    - one timestamped output folder per instance.
    """
    dataset = load_dataset()
    instances = dataset["instances"]

    if args.list_instances:
        print("Available instances:")
        print(", ".join(sorted(instances)))
        return None

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")

    args.benchmarks = normalize_benchmarks(getattr(args, "benchmarks", None))

    instance_names = select_instance_names(args, instances)
    custom_c_values = normalize_custom_c_values(args.c_values)

    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = PROJECT_ROOT / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = timestamp_string()

    instance_records = {}
    jobs_data_by_instance = {}
    c_values_by_instance = {}
    output_dirs_by_instance = {}
    cpsat_opt_refs = {}
    tn_results_by_instance = {}

    print("=" * 70)
    print("JSSP-QUBO multi-instance benchmark")
    print("=" * 70)
    print(f"Selected instances: {', '.join(instance_names)}")
    print(
        "Selected benchmarks: "
        + ", ".join(BENCHMARK_LABELS[name] for name in args.benchmarks)
    )
    print(f"Total worker processes requested: {args.max_workers}")
    print(f"Run timestamp: {run_timestamp}")
    print()

    for instance_name in instance_names:
        instance_record = instances[instance_name]
        jobs_data = normalize_jobs_data(instance_record)

        instance_records[instance_name] = instance_record
        jobs_data_by_instance[instance_name] = jobs_data

        print_instance_summary(instance_name, instance_record)

        if "cpsat" not in args.benchmarks:
            cpsat_opt_ref = {
                "status": "SKIPPED",
                "makespan": None,
                "time": None,
            }
            print("Skipping CP-SAT optimization reference because CP-SAT is not selected.")
            print()
        elif args.skip_cpsat_optimize:
            cpsat_opt_ref = {
                "status": "SKIPPED",
                "makespan": None,
                "time": None,
            }
            print("Skipping extra CP-SAT optimization reference.")
            print(
                "Note: CP-SAT fixed-C feasibility still runs inside each C benchmark."
            )
            print()
        else:
            print(f"Running CP-SAT optimization reference for {instance_name}...")
            opt_ref = solve_jsp_cpsat_optimize(
                jobs_data,
                time_limit=args.cpsat_time_limit,
            )

            cpsat_opt_ref = {
                "status": opt_ref["status"],
                "makespan": opt_ref["makespan"],
                "time": opt_ref["time"],
            }

            print("CP-SAT optimization reference:")
            print(f"  status: {opt_ref['status']}")
            print(f"  makespan: {opt_ref['makespan']}")
            print(f"  time_sec: {opt_ref['time']:.4f}")
            print()

        cpsat_opt_refs[instance_name] = cpsat_opt_ref

        if "tn_lns" in args.benchmarks:
            print(f"Running TN-LNS reference for {instance_name}...")
            tn_result = solve_jsp_tn_lns(
                jobs_data,
                optimum=instance_record.get("optimum"),
                seed=args.seed,
            )
            tn_results_by_instance[instance_name] = tn_result
            print("TN-LNS reference:")
            print(f"  status: {tn_result['status']}")
            print(f"  makespan: {tn_result['makespan']}")
            print(f"  time_sec: {tn_result['time']:.4f}")
            print()
        else:
            tn_results_by_instance[instance_name] = None

        if custom_c_values is not None:
            c_values = custom_c_values
        else:
            c_values = choose_c_values(
                instance_record=instance_record,
                opt_makespan=cpsat_opt_ref["makespan"],
            )

        c_values_by_instance[instance_name] = c_values

        output_dir = make_run_output_dir(
            base_output_dir=base_output_dir,
            instance_name=instance_name,
            run_timestamp=run_timestamp,
        )
        output_dirs_by_instance[instance_name] = output_dir

        save_run_metadata(
            output_dir=output_dir,
            instance_name=instance_name,
            instance_record=instance_record,
            c_values=c_values,
            args=args,
            run_timestamp=run_timestamp,
            cpsat_opt_ref=cpsat_opt_ref,
        )

        print(f"C-values for {instance_name}: {c_values}")
        print(f"Output directory for {instance_name}: {output_dir}")
        print()

    tasks = []

    for instance_name in instance_names:
        for C in c_values_by_instance[instance_name]:
            tasks.append(
                {
                    "instance_name": instance_name,
                    "jobs_data": jobs_data_by_instance[instance_name],
                    "C": C,
                    "num_reads": args.num_reads,
                    "num_sweeps": args.num_sweeps,
                    "seed": args.seed,
                    "cpsat_time_limit": args.cpsat_time_limit,
                    "benchmarks": args.benchmarks,
                    "known_optimum": instance_records[instance_name].get("optimum"),
                    "tn_result": tn_results_by_instance.get(instance_name),
                }
            )

    print("=" * 70)
    print("Task plan")
    print("=" * 70)
    print(f"Total fixed-C tasks: {len(tasks)}")
    print(
        "Each task is one independent benchmark for one pair "
        "(instance, fixed makespan C)."
    )

    actual_workers = min(args.max_workers, len(tasks))

    if actual_workers <= 1:
        print("Execution mode: sequential")
    else:
        print(f"Execution mode: parallel with {actual_workers} worker processes")

    print()

    completed = {}

    if actual_workers <= 1:
        for task in tasks:
            instance_name = task["instance_name"]
            C = task["C"]
            print(f"Running task: instance={instance_name}, C={C}")

            result_instance_name, result_C, df = run_one_instance_c_task(task)
            completed[(result_instance_name, result_C)] = df
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_task = {
                executor.submit(run_one_instance_c_task, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                instance_name = task["instance_name"]
                C = task["C"]

                try:
                    result_instance_name, result_C, df = future.result()
                except Exception as exc:
                    print(f"ERROR in task instance={instance_name}, C={C}: {exc}")
                    raise

                completed[(result_instance_name, result_C)] = df
                print(f"Finished task: instance={result_instance_name}, C={result_C}")

    all_result_frames = []

    for instance_name in instance_names:
        instance_record = instance_records[instance_name]
        output_dir = output_dirs_by_instance[instance_name]
        c_values = c_values_by_instance[instance_name]

        instance_frames = []

        for C in c_values:
            df = completed[(instance_name, C)]
            df = add_result_metadata(
                df=df,
                instance_name=instance_name,
                instance_record=instance_record,
                output_dir=output_dir,
                run_timestamp=run_timestamp,
            )
            instance_frames.append(df)

        instance_results_df = pd.concat(instance_frames, ignore_index=True)

        save_outputs(
            results_df=instance_results_df,
            c_values=c_values,
            output_dir=output_dir,
            instance_name=instance_name,
        )

        all_result_frames.append(instance_results_df)

    combined_results_df = pd.concat(all_result_frames, ignore_index=True)

    batch_results_path = base_output_dir / f"batch-{run_timestamp}_all_results.csv"
    combined_results_df.to_csv(batch_results_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print()
    print("=" * 70)
    print("Combined benchmark summary")
    print("=" * 70)
    print(combined_results_df)
    print()
    print(f"Saved combined batch results to: {batch_results_path}")

    return combined_results_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run JSP-QUBO benchmarks on one or more named OR-Library/JSPLib instances."
        )
    )

    parser.add_argument(
        "--instances",
        "--instance",
        dest="instances",
        nargs="+",
        default=None,
        help=(
            "Instance name(s), e.g. ft06 or ft06 la01 abz5. "
            "If omitted, prompts interactively."
        ),
    )

    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="List available instance names and exit.",
    )

    parser.add_argument(
        "--c-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Fixed makespan values C to benchmark. "
            "If omitted, values are chosen separately for each instance "
            "from optimum/lower/upper bounds."
        ),
    )

    parser.add_argument(
        "--num-reads",
        type=int,
        default=100,
        help="Number of simulated annealing reads.",
    )

    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=1000,
        help="Number of simulated annealing sweeps per read.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for simulated annealing.",
    )

    parser.add_argument(
        "--cpsat-time-limit",
        type=float,
        default=5.0,
        help="CP-SAT time limit in seconds.",
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Benchmarks to run. Default: all. "
            "Use any of: cpsat, time_indexed_qubo, compact_disjunctive_qubo, "
            "tn_lns. Aliases such as classical, ti, compact, and tn are accepted."
        ),
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Total number of parallel worker processes to use across all "
            "(instance, C) tasks. Example: 12 workers for 2 instances with "
            "6 C-values each."
        ),
    )

    parser.add_argument(
        "--skip-cpsat-optimize",
        action="store_true",
        help=(
            "Skip the extra CP-SAT optimization reference before fixed-C runs. "
            "The CP-SAT fixed-C feasibility row inside benchmark_fixed_C still runs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Base directory where benchmark outputs will be written. "
            "Each instance gets a subfolder like outputs/ft06-20260608223315."
        ),
    )

    args = parser.parse_args()
    run_instance_benchmark(args)


if __name__ == "__main__":
    main()
