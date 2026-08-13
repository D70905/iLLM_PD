# -*- coding: utf-8 -*-
"""Run alpha sensitivity training cells for R2.5/R3.2.

This launcher only trains policies. After training, run scripts/ltpp_inference.py
with --enable-llm for the full-system delivered-design evaluation.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURROGATE_MODEL = "./output/surrogate_model/surrogate_v3.pt"

ALPHA_CELLS = {
    "pure_ppo": {"no_llm": True, "alpha": 0.0, "fallback": 0.0},
    "alpha_0p3": {"no_llm": False, "alpha": 0.3, "fallback": 0.0},
    "alpha_0p5": {"no_llm": False, "alpha": 0.5, "fallback": 0.0},
    "alpha_0p7": {"no_llm": False, "alpha": 0.7, "fallback": 0.0},
    "alpha_old_fallback": {"no_llm": False, "alpha": 0.5, "fallback": 0.8},
}


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def pyexe() -> str:
    candidate = Path(sys.executable).with_name("python.exe")
    return str(candidate if candidate.exists() else Path(sys.executable))


def build_command(cell: str, pavement: str, seed: int, timesteps: int,
                  gen_backend: str, gen_model: str | None, b3_threshold: float,
                  use_reranker: bool, force_name: str | None = None) -> list[str]:
    spec = ALPHA_CELLS[cell]
    run_name = force_name or f"sens_alpha_{cell}_{pavement}_{timesteps}ts_seed{seed}"
    cmd = [
        pyexe(), "-m", "rl.train",
        "--pavement-type", pavement,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--use-surrogate",
        "--surrogate-model-path", SURROGATE_MODEL,
        "--surrogate-b3-threshold", str(b3_threshold),
        "--run-name", run_name,
        "--gen-backend", gen_backend,
        "--gen-alpha-fallback", str(spec["fallback"]),
    ]
    if spec["no_llm"]:
        cmd.append("--no-llm")
    else:
        if gen_model:
            cmd += ["--gen-model", gen_model]
        if not use_reranker:
            cmd.append("--gen-no-reranker")
        cmd += ["--gen-alpha-initial", str(spec["alpha"]), "--gen-alpha-decay", "linear_to_zero"]
    return cmd


def is_complete(run_name: str) -> bool:
    run_dir = PROJECT_ROOT / "output" / "rl_runs" / run_name
    if (run_dir / "training_complete.flag").exists():
        return True
    ckpt_root = run_dir / "checkpoints"
    return ckpt_root.exists() and any((d / "ppo_model.zip").exists() for d in ckpt_root.glob("ckpt_final_step_*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha sensitivity training launcher")
    parser.add_argument("--cells", default="all", help="Comma cells or 'all'.")
    parser.add_argument("--pavement", default="flexible", choices=["flexible", "semi_rigid", "both"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--gen-backend", default="deepseek", choices=["deepseek", "chatfire", "siliconflow-qwen", "siliconflow-glm", "ollama", "ollama-llama"])
    parser.add_argument("--gen-model", default="deepseek-chat",
                        help="Structured-output model used for alpha sensitivity.")
    parser.add_argument("--surrogate-b3-threshold", type=float, default=0.7,
                        help="Training-time B3 escalation threshold; use 1.0 for stricter but slower training.")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Enable LLM RAG reranker during training. Default off for speed/cost.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cells = list(ALPHA_CELLS) if args.cells == "all" else [x.strip() for x in args.cells.split(",") if x.strip()]
    pavements = ["flexible", "semi_rigid"] if args.pavement == "both" else [args.pavement]
    seeds = parse_seeds(args.seeds)

    queue = []
    for cell in cells:
        if cell not in ALPHA_CELLS:
            raise SystemExit(f"Unknown alpha cell: {cell}")
        for pavement in pavements:
            for seed in seeds:
                run_name = f"sens_alpha_{cell}_{pavement}_{args.timesteps}ts_seed{seed}"
                if not args.force and is_complete(run_name):
                    print(f"[skip] {run_name}")
                    continue
                queue.append((cell, pavement, seed, run_name))

    print(f"Alpha sensitivity queue: {len(queue)} runs")
    for i, (cell, pavement, seed, run_name) in enumerate(queue, 1):
        cmd = build_command(cell, pavement, seed, args.timesteps,
                            args.gen_backend, args.gen_model,
                            args.surrogate_b3_threshold, args.use_reranker,
                            force_name=run_name)
        print(f"[{i}/{len(queue)}] {run_name}")
        print("  " + " ".join(f'\"{c}\"' if " " in c else c for c in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()



