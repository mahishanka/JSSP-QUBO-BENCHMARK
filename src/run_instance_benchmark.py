import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from jsp_qubo_benchmark import (
    benchmark_fixed_C,
    compute_reduction_ratios,
    save_comparison_plot,
    solve_jsp_cpsat_optimize,
)


# Since this file lives in:
#   JSSP-QUBO-BENCHMARK/src/run_instance_benchmark.py
#
# PROJECT_ROOT should be:
#   JSSP-QUBO-BENCHMARK/
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

    Format:
        YYYYMMDDHHMMSS
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def load_dataset(path: Path = DATASET_PATH) -> dict:
    """
    Load the parsed OR-Library + bounds JSON file.

    Expected top-level structure:
        {
            "schema_version": "...",
            "instances": {
                "ft06": {...},
                "abz5": {...},
                ...
            }
        }
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


def make_run_output_dir(base_output_dir: Path, instance_name: str) -> Path:
    """
    Create a unique output directory for this run.

    Example:
        outputs/abz5-20260608223315
    """
    run_name = f"{instance_name}-{timestamp_string()}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_run_metadata(
    output_dir: Path,
    instance_name: str,
    instance_record: dict,
    c_values: list[int],
    args,
):
    """
    Save a small JSON file describing this benchmark run.
    """
    metadata = {
        "instance": instance_name,
        "timestamp": output_dir.name.split("-", maxsplit=1)[-1],
        "output_dir": str(output_dir),
        "c_values": c_values,
        "run_settings": {
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "seed": args.seed,
            "cpsat_time_limit": args.cpsat_time_limit,
        },
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
    Save CSV outputs and comparison plots.
    """
    results_path = output_dir / f"{instance_name}_jsp_qubo_benchmark_results.csv"
    ratios_path = output_dir / f"{instance_name}_jsp_qubo_reduction_ratios.csv"

    results_df.to_csv(results_path, index=False)

    ratios_df = compute_reduction_ratios(results_df, c_values)
    ratios_df.insert(0, "instance", instance_name)
    ratios_df.to_csv(ratios_path, index=False)

    qubo_results = results_df[
        results_df["model"].isin(
            ["Time-indexed QUBO", "Compact disjunctive QUBO"]
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
        qubo_results,
        "time_sec",
        "Runtime (s)",
        f"{instance_name}: QUBO runtime comparison",
        output_dir / f"{instance_name}_qubo_runtime_comparison.png",
    )

    print()
    print("Saved outputs:")
    print(f"  {results_path}")
    print(f"  {ratios_path}")
    print(f"  {output_dir / f'{instance_name}_qubo_variable_count_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_qubo_quadratic_terms_comparison.png'}")
    print(f"  {output_dir / f'{instance_name}_qubo_runtime_comparison.png'}")


def print_instance_summary(instance_name: str, instance_record: dict):
    """
    Print useful metadata for the selected instance.
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


def run_instance_benchmark(args):
    """
    Main wrapper logic.

    This function:
    1. Loads the JSON dataset.
    2. Selects a named instance.
    3. Converts jobs_data to Mahi's tuple-pair format.
    4. Creates a timestamped output directory.
    5. Runs CP-SAT optimization as a reference.
    6. Chooses or accepts fixed C values.
    7. Calls Mahi's existing benchmark_fixed_C function.
    8. Saves CSV, plots, and run metadata under outputs/<instance>-<timestamp>/.
    """
    dataset = load_dataset()
    instances = dataset["instances"]

    if args.list_instances:
        print("Available instances:")
        print(", ".join(sorted(instances)))
        return None

    if args.instance is None:
        print("Available instances:")
        print(", ".join(sorted(instances)))
        instance_name = input("Which instance do you want to run? ").strip().lower()
    else:
        instance_name = args.instance.strip().lower()

    if instance_name not in instances:
        names_preview = ", ".join(sorted(instances)[:30])
        raise ValueError(
            f"Unknown instance '{instance_name}'.\n"
            f"First available names include:\n"
            f"  {names_preview}, ..."
        )

    instance_record = instances[instance_name]
    jobs_data = normalize_jobs_data(instance_record)

    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = PROJECT_ROOT / base_output_dir

    output_dir = make_run_output_dir(
        base_output_dir=base_output_dir,
        instance_name=instance_name,
    )

    print_instance_summary(instance_name, instance_record)

    print("Run settings:")
    print(f"  num_reads: {args.num_reads}")
    print(f"  num_sweeps: {args.num_sweeps}")
    print(f"  seed: {args.seed}")
    print(f"  cpsat_time_limit: {args.cpsat_time_limit}")
    print(f"  output_dir: {output_dir}")
    print()

    print("Running CP-SAT optimization reference...")
    opt_ref = solve_jsp_cpsat_optimize(
        jobs_data,
        time_limit=args.cpsat_time_limit,
    )

    print("CP-SAT optimization reference:")
    print(f"  status: {opt_ref['status']}")
    print(f"  makespan: {opt_ref['makespan']}")
    print(f"  time_sec: {opt_ref['time']:.4f}")
    print()

    if args.c_values is not None:
        c_values = args.c_values
    else:
        c_values = choose_c_values(
            instance_record=instance_record,
            opt_makespan=opt_ref["makespan"],
        )

    print(f"Testing fixed C values: {c_values}")
    print()

    save_run_metadata(
        output_dir=output_dir,
        instance_name=instance_name,
        instance_record=instance_record,
        c_values=c_values,
        args=args,
    )

    result_frames = []

    for C in c_values:
        print(f"Running fixed-C benchmark for C={C}...")

        df = benchmark_fixed_C(
            jobs_data,
            C,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
            cpsat_time_limit=args.cpsat_time_limit,
        )

        df.insert(0, "instance", instance_name)
        df.insert(1, "known_lower_bound", instance_record.get("lower_bound"))
        df.insert(2, "known_upper_bound", instance_record.get("upper_bound"))
        df.insert(3, "known_optimum", instance_record.get("optimum"))
        df.insert(4, "known_status", instance_record.get("status"))
        df.insert(5, "run_output_dir", str(output_dir))

        result_frames.append(df)

    results_df = pd.concat(result_frames, ignore_index=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print()
    print("Benchmark summary:")
    print(results_df)

    save_outputs(
        results_df=results_df,
        c_values=c_values,
        output_dir=output_dir,
        instance_name=instance_name,
    )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run JSP-QUBO benchmark on a named OR-Library/JSPLib instance."
    )

    parser.add_argument(
        "--instance",
        default=None,
        help=(
            "Instance name, e.g. ft06, ft10, la01, abz5. "
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
            "If omitted, values are chosen from optimum/lower/upper bounds."
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
        "--output-dir",
        default="outputs",
        help=(
            "Base directory where benchmark outputs will be written. "
            "Each run gets a subfolder like outputs/ft06-20260608223315."
        ),
    )

    args = parser.parse_args()
    run_instance_benchmark(args)


if __name__ == "__main__":
    main()