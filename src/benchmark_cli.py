import os
from argparse import Namespace

from jsp_qubo_benchmark import BENCHMARK_LABELS, DEFAULT_BENCHMARKS, normalize_benchmarks
from run_instance_benchmark import load_dataset, run_instance_benchmark


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    Ask a yes/no question.
    """
    default_text = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{prompt} [{default_text}]: ").strip().lower()

        if answer == "":
            return default

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    """
    Ask for an integer, with a default value.
    """
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()

        if answer == "":
            value = default
        else:
            try:
                value = int(answer)
            except ValueError:
                print("Please enter an integer.")
                continue

        if minimum is not None and value < minimum:
            print(f"Please enter an integer at least {minimum}.")
            continue

        return value


def ask_float(prompt: str, default: float, minimum: float | None = None) -> float:
    """
    Ask for a float, with a default value.
    """
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()

        if answer == "":
            value = default
        else:
            try:
                value = float(answer)
            except ValueError:
                print("Please enter a number.")
                continue

        if minimum is not None and value < minimum:
            print(f"Please enter a number at least {minimum}.")
            continue

        return value


def parse_instance_names(raw: str) -> list[str]:
    """
    Accept either spaces or commas:
        ft06 la01
        ft06, la01
    """
    cleaned = raw.replace(",", " ")
    return [x.strip().lower() for x in cleaned.split() if x.strip()]


def ask_instance_names(instances: dict) -> list[str]:
    """
    Ask the user which instances to run.
    """
    names = sorted(instances)

    print()
    print("Available instance examples:")
    print("  ft06, ft10, ft20")
    print("  la01, la02, ..., la40")
    print("  abz5, abz6, ..., abz9")
    print("  orb01, ..., orb10")
    print("  swv01, ..., swv20")
    print("  yn1, ..., yn4")
    print()

    show_all = ask_yes_no("Show all available instance names?", default=False)

    if show_all:
        print()
        print(", ".join(names))
        print()

    while True:
        raw = input(
            "Which instance(s) do you want to run? "
            "Use spaces or commas, e.g. ft06 la01: "
        ).strip()

        instance_names = parse_instance_names(raw)

        if not instance_names:
            print("Please enter at least one instance name.")
            continue

        unknown = [name for name in instance_names if name not in instances]

        if not unknown:
            return list(dict.fromkeys(instance_names))

        print(f"Unknown instance(s): {', '.join(unknown)}")
        print("Try something like ft06, la01, abz5, orb01, swv01, or yn1.")


def ask_run_mode():
    """
    Ask for solver settings.

    Returns:
        num_reads, num_sweeps, cpsat_time_limit
    """
    print()
    print("Choose run mode:")
    print("  1. Quick smoke test    -- fast, just checks that the code works")
    print("  2. Normal test         -- default benchmark settings")
    print("  3. Custom settings")
    print()

    while True:
        choice = input("Run mode [1]: ").strip()

        if choice == "":
            choice = "1"

        if choice == "1":
            return {
                "num_reads": 10,
                "num_sweeps": 100,
                "cpsat_time_limit": 2.0,
            }

        if choice == "2":
            return {
                "num_reads": 100,
                "num_sweeps": 1000,
                "cpsat_time_limit": 5.0,
            }

        if choice == "3":
            return {
                "num_reads": ask_int(
                    "Number of simulated annealing reads",
                    100,
                    minimum=1,
                ),
                "num_sweeps": ask_int(
                    "Number of simulated annealing sweeps",
                    1000,
                    minimum=1,
                ),
                "cpsat_time_limit": ask_float(
                    "CP-SAT time limit in seconds",
                    5.0,
                    minimum=0.0,
                ),
            }

        print("Please choose 1, 2, or 3.")


def ask_c_values():
    """
    Ask whether to use automatic C-values or custom C-values.

    Custom C-values apply to every selected instance.
    """
    use_auto = ask_yes_no(
        "Use automatic C-values from known optimum/lower/upper bounds?",
        default=True,
    )

    if use_auto:
        return None

    print()
    print("Custom C-values will be used for every selected instance.")

    while True:
        answer = input(
            "Enter C-values separated by spaces, e.g. 53 54 55 56 58 60: "
        ).strip()

        try:
            values = [int(x) for x in answer.split()]
        except ValueError:
            print("Please enter only integers separated by spaces.")
            continue

        if not values:
            print("Please enter at least one C-value.")
            continue

        return list(dict.fromkeys(values))


def ask_benchmarks() -> list[str]:
    """
    Ask which benchmark methods to run.
    """
    print()
    print("Benchmark methods:")
    for index, name in enumerate(DEFAULT_BENCHMARKS, start=1):
        print(f"  {index}. {BENCHMARK_LABELS[name]}")
    print()
    print("Press Enter for all, or enter numbers/names separated by spaces.")
    print("Examples: 1 4, cpsat tn, ti compact")

    number_to_name = {
        str(index): name
        for index, name in enumerate(DEFAULT_BENCHMARKS, start=1)
    }

    while True:
        raw = input("Benchmarks to run [all]: ").strip()

        if raw == "":
            return list(DEFAULT_BENCHMARKS)

        tokens = raw.replace(",", " ").split()
        requested = [number_to_name.get(token, token) for token in tokens]

        try:
            return normalize_benchmarks(requested)
        except ValueError as exc:
            print(exc)


def ask_max_workers() -> int:
    """
    Ask how many total worker processes to use.
    """
    cpu_count = os.cpu_count() or 1

    print()
    print("Parallelism setting:")
    print(
        "The program parallelizes over all selected (instance, C) pairs. "
        "For example, 2 instances with 6 C-values each gives 12 independent tasks."
    )
    print(f"Detected logical CPUs: {cpu_count}")
    print("Use 1 for sequential execution.")
    print("For large instances, using too many workers can run out of memory.")
    print()

    return ask_int(
        "Total worker processes to use",
        default=1,
        minimum=1,
    )


def main():
    """
    Interactive CLI wrapper.

    This script asks for input interactively and then calls the existing
    run_instance_benchmark(args) function.
    """
    print("=" * 70)
    print("JSSP-QUBO Benchmark CLI")
    print("=" * 70)

    dataset = load_dataset()
    instances = dataset["instances"]

    instance_names = ask_instance_names(instances)
    benchmarks = ask_benchmarks()
    run_settings = ask_run_mode()
    c_values = ask_c_values()
    max_workers = ask_max_workers()
    seed = ask_int("Random seed", 1)

    if "cpsat" in benchmarks:
        run_extra_cpsat_opt = ask_yes_no(
            "Run extra CP-SAT optimization reference before fixed-C benchmarks?",
            default=True,
        )
    else:
        run_extra_cpsat_opt = False

    print()
    print("=" * 70)
    print("Run summary")
    print("=" * 70)
    print(f"Instances: {', '.join(instance_names)}")
    print(
        "Benchmarks: "
        + ", ".join(BENCHMARK_LABELS[name] for name in benchmarks)
    )
    print(f"C-values: {'automatic per instance' if c_values is None else c_values}")
    print(f"num_reads: {run_settings['num_reads']}")
    print(f"num_sweeps: {run_settings['num_sweeps']}")
    print(f"cpsat_time_limit: {run_settings['cpsat_time_limit']}")
    print(f"max_workers: {max_workers}")
    print(f"seed: {seed}")
    print(f"extra CP-SAT optimize reference: {run_extra_cpsat_opt}")
    print("Output folders: outputs/<instance>-<timestamp>/")
    print()

    proceed = ask_yes_no("Start benchmark?", default=True)

    if not proceed:
        print("Cancelled.")
        return

    args = Namespace(
        instances=instance_names,
        list_instances=False,
        benchmarks=benchmarks,
        c_values=c_values,
        num_reads=run_settings["num_reads"],
        num_sweeps=run_settings["num_sweeps"],
        seed=seed,
        cpsat_time_limit=run_settings["cpsat_time_limit"],
        max_workers=max_workers,
        skip_cpsat_optimize=not run_extra_cpsat_opt,
        output_dir="outputs",
    )

    run_instance_benchmark(args)


if __name__ == "__main__":
    main()
