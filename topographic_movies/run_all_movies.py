"""Run all topographic movie configs in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def list_configs(pattern: str | None = None) -> list[Path]:
    configs = [cfg for cfg in sorted(CONFIG_DIR.glob("*.yaml")) if cfg.name != "common.yaml"]
    if pattern:
        configs = [cfg for cfg in configs if pattern in cfg.name]
    return configs


def run_config(config_path: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "code.run_topo_movie", "--config", str(config_path)]
    cmd.extend(extra_args)
    print(f"\n=== Running {config_path.name} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Run failed for {config_path}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all topographic movie configurations")
    parser.add_argument("--pattern", help="Optional substring filter for config filenames")
    parser.add_argument("--list", action="store_true", help="List matching configs without running")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional arguments passed to run_topo_movie")
    args = parser.parse_args()

    configs = list_configs(args.pattern)
    if not configs:
        print("No configs matched.")
        return 1

    if args.list:
        for cfg in configs:
            print(cfg)
        return 0

    failures = 0
    for cfg in configs:
        rc = run_config(cfg, args.extra)
        if rc != 0:
            failures += 1

    if failures:
        print(f"{failures} run(s) failed.")
        return 1

    print("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

