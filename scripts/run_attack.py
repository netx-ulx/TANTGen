"""Universal TANTGen attack runner.

Reads `scripts/attacks.yaml` and dispatches to the appropriate pipeline in
`scripts/utils.py`. Use this in place of the legacy `scripts/run_<attack>.py`
files for any new attack added to the registry.

    # single attack
    python -m scripts.run_attack Mirai

    # all attacks in one family (still serial; each pipeline takes hours)
    python -m scripts.run_attack --all-kitsune
    python -m scripts.run_attack --all-cicids
    python -m scripts.run_attack --all-cicddos
    python -m scripts.run_attack --all-hypervision
    python -m scripts.run_attack --all

    # inspection
    python -m scripts.run_attack --list
    python -m scripts.run_attack Mirai --dry-run

NOTE: `mean_std.json` is overwritten between attacks (same behavior as the
legacy per-attack scripts). Batches are serial so this is consistent within
each pipeline run.
"""

import argparse
import os
import sys
import time

import yaml

from scripts.utils import (
    run_kitsune_on_kitsune_pipeline,
    run_ml_nids_on_cicids_pipeline,
    run_kitsune_on_CICIDS_pipeline,
    run_hypervision_pipeline,
)


REGISTRY_PATH = "scripts/attacks.yaml"
FINETUNE_CONFIG_PATH = "src/TANTGen/configs/config-TANTGen-finetune.yaml"

FAMILY_FLAGS = {
    "--all-kitsune": "kitsune",
    "--all-cicids": "cicids",
    "--all-cicddos": "cicddos",
    "--all-hypervision": "hypervision",
}


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        sys.exit(f"Registry not found: {REGISTRY_PATH}")
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def template_finetune_margin(margin):
    """Rewrite the `margin` line in config-TANTGen-finetune.yaml in place.

    Preserves every other line as-is (epochs/dataset_path/etc.). The TANTGen
    process re-reads this file on its next `subprocess.run()` invocation, so
    the next finetune cycle picks up the per-attack margin.
    """
    if margin is None:
        return
    with open(FINETUNE_CONFIG_PATH) as f:
        lines = f.readlines()
    new_lines = []
    found = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("margin:") or stripped.startswith("margin :"):
            indent = line[: len(line) - len(stripped)]
            new_lines.append(f"{indent}margin: {margin}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"margin: {margin}\n")
    with open(FINETUNE_CONFIG_PATH, "w") as f:
        f.writelines(new_lines)


def dispatch(attack_id, cfg, dry_run=False):
    pipeline = cfg["pipeline"]
    pcap = cfg["pcap"]
    labels = cfg["labels"]
    ratios = cfg["ratios"]
    margin = cfg.get("margin")

    print("=" * 70)
    print(f"[run_attack] attack={attack_id}  pipeline={pipeline}  margin={margin}")
    print(f"[run_attack] pcap   = {pcap}")
    print(f"[run_attack] labels = {labels}")
    print("=" * 70)

    if dry_run:
        return

    for required in (pcap, labels):
        if not os.path.exists(required):
            sys.exit(f"[run_attack] Missing input file: {required}")

    template_finetune_margin(margin)

    if pipeline == "kitsune_native":
        run_kitsune_on_kitsune_pipeline(
            attack_id, pcap, labels,
            ratios["train_nids"], ratios["train_surrogate_b"],
            ratios["train_adversarial"], ratios["train_surrogate_m"],
        )
    elif pipeline == "cicids_ml":
        run_ml_nids_on_cicids_pipeline(
            attack_id, pcap, labels,
            ratios["nids"], ratios["surrogate"],
            ratios["adversarial"], ratios["test"],
        )
    elif pipeline == "cicids_kitsune":
        run_kitsune_on_CICIDS_pipeline(
            attack_id, pcap, labels,
            ratios["train_nids"], ratios["train_surrogate_b"],
            ratios["train_adversarial"], ratios["train_surrogate_m"],
        )
    elif pipeline == "hypervision":
        run_hypervision_pipeline(
            attack_id, pcap, labels,
            ratios["train_nids"], ratios["train_surrogate_b"],
            ratios["train_adversarial"], ratios["train_surrogate_m"],
        )
    else:
        sys.exit(f"[run_attack] Unknown pipeline '{pipeline}' for attack '{attack_id}'")


def print_listing(registry):
    width = max(len(k) for k in registry)
    print(f"{'attack_id'.ljust(width)}  family        pipeline           pcap")
    print(f"{'-' * width}  ------------  -----------------  ----")
    for name, cfg in registry.items():
        print(f"{name.ljust(width)}  "
              f"{cfg.get('family', '-').ljust(12)}  "
              f"{cfg['pipeline'].ljust(17)}  "
              f"{cfg['pcap']}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal TANTGen attack runner (replaces the legacy scripts/run_*.py)."
    )
    parser.add_argument("attack", nargs="?",
                        help="Attack id from scripts/attacks.yaml (omit when using --all* or --list).")
    parser.add_argument("--all", action="store_true",
                        help="Run every attack in the registry, serially.")
    parser.add_argument("--all-kitsune", action="store_true",
                        help="Run every attack with family=kitsune.")
    parser.add_argument("--all-cicids", action="store_true",
                        help="Run every attack with family=cicids.")
    parser.add_argument("--all-cicddos", action="store_true",
                        help="Run every attack with family=cicddos.")
    parser.add_argument("--all-hypervision", action="store_true",
                        help="Run every attack with family=hypervision.")
    parser.add_argument("--list", action="store_true",
                        help="Print the registry and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run, but don't invoke the pipeline.")
    args = parser.parse_args()

    registry = load_registry()

    if args.list:
        print_listing(registry)
        return

    selected = []
    if args.attack:
        if args.attack not in registry:
            sys.exit(f"Unknown attack '{args.attack}'. Run with --list to see options.")
        selected.append(args.attack)
    if args.all:
        selected.extend([k for k in registry if k not in selected])
    for flag, family in FAMILY_FLAGS.items():
        attr = flag.lstrip("-").replace("-", "_")
        if getattr(args, attr):
            selected.extend([k for k, cfg in registry.items()
                             if cfg.get("family") == family and k not in selected])

    if not selected:
        parser.print_help()
        sys.exit("\nNo attack selected. Pass an attack id or one of --all* / --list.")

    print(f"[run_attack] dispatching {len(selected)} attack(s): {', '.join(selected)}")
    start = time.time()
    for name in selected:
        attack_start = time.time()
        try:
            dispatch(name, registry[name], dry_run=args.dry_run)
        except KeyboardInterrupt:
            print(f"\n[run_attack] interrupted during '{name}'")
            raise
        print(f"[run_attack] '{name}' finished in {time.time() - attack_start:.1f}s")
    print(f"[run_attack] all done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
