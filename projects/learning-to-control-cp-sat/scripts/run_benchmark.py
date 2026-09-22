from __future__ import annotations

import argparse

from learning_cp_sat.experiment import load_config, run_experiment, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the learning-to-control CP benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_experiment(load_config(args.config))
    write_result(result, args.output)
    selected = result["learning"]["selected_model"]
    val_acc = result["learning"]["validation_top1_imitation_accuracy"][selected]
    print(f"selected_model={selected} validation_top1_accuracy={val_acc:.3f}")
    print(f"result={args.output}")


if __name__ == "__main__":
    main()
