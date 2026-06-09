from argparse import Namespace

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


def ask_int(prompt: str, default: int) -> int:
    """
    Ask for an integer, with a default value.
    """
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()

        if answer == "":
            return default

        try:
            return int(answer)
        except ValueError:
            print("Please enter an integer.")


def ask_float(prompt: str, default: float) -> float:
    """
    Ask for a float, with a default value.
    """
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()

        if answer == "":
            return default

        try:
            return float(answer)
        except ValueError:
            print("Please enter a number.")


def ask_c_values():
    """
    Ask whether to use automatic C-values or custom C-values.
    """
    use_auto = ask_yes_no(
        "Use automatic C-values from known optimum/lower/upper bounds?",
        default=True,
    )

    if use_auto:
        return None

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

        return values


def ask_instance_name(instances: dict) -> str:
    """
    Ask the user which instance to run.
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
        instance_name = input("Which instance do you want to run? ").strip().lower()

        if instance_name in instances:
            return instance_name

        print(f"Unknown instance: {instance_name}")
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
                "num_reads": ask_int("Number of simulated annealing reads", 100),
                "num_sweeps": ask_int("Number of simulated annealing sweeps", 1000),
                "cpsat_time_limit": ask_float("CP-SAT time limit in seconds", 5.0),
            }

        print("Please choose 1, 2, or 3.")


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

    instance_name = ask_instance_name(instances)
    run_settings = ask_run_mode()
    c_values = ask_c_values()
    seed = ask_int("Random seed", 1)

    print()
    print("=" * 70)
    print("Run summary")
    print("=" * 70)
    print(f"Instance: {instance_name}")
    print(f"C-values: {'automatic' if c_values is None else c_values}")
    print(f"num_reads: {run_settings['num_reads']}")
    print(f"num_sweeps: {run_settings['num_sweeps']}")
    print(f"cpsat_time_limit: {run_settings['cpsat_time_limit']}")
    print(f"seed: {seed}")
    print("Output folder: outputs/<instance>-<timestamp>/")
    print()

    proceed = ask_yes_no("Start benchmark?", default=True)

    if not proceed:
        print("Cancelled.")
        return

    args = Namespace(
        instance=instance_name,
        list_instances=False,
        c_values=c_values,
        num_reads=run_settings["num_reads"],
        num_sweeps=run_settings["num_sweeps"],
        seed=seed,
        cpsat_time_limit=run_settings["cpsat_time_limit"],
        output_dir="outputs",
    )

    run_instance_benchmark(args)


if __name__ == "__main__":
    main()