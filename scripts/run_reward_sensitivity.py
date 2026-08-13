# -*- coding: utf-8 -*-
"""Run reward-weight sensitivity training cells for R3.4."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURROGATE_MODEL = "./output/surrogate_model/surrogate_v3.pt"
REWARD_PROFILES = ["baseline", "performance-up-12p5", "performance-up", "performance-up-37p5", "performance-up-50p0", "economy-up", "economy-down", "no-directional-reward", "performance-down"]


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def pyexe() -> str:
    candidate = Path(sys.executable).with_name("python.exe")
    return str(candidate if candidate.exists() else Path(sys.executable))


def run_name(profile: str, pavement: str, timesteps: int, seed: int) -> str:
    return f"sens_reward_{profile}_{pavement}_{timesteps}ts_seed{seed}"


def is_complete(name: str) -> bool:
    run_dir = PROJECT_ROOT / "output" / "rl_runs" / name
    if (run_dir / "training_complete.flag").exists():
        return True
    ckpt_root = run_dir / "checkpoints"
    return ckpt_root.exists() and any((d / "ppo_model.zip").exists() for d in ckpt_root.glob("ckpt_final_step_*"))


def build_command(profile: str, pavement: str, seed: int, timesteps: int, gen_backend: str, gen_model: str | None, b3_threshold: float, use_reranker: bool) -> list[str]:
    cmd = [
        pyexe(), "-m", "rl.train",
        "--pavement-type", pavement,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--use-surrogate",
        "--surrogate-model-path", SURROGATE_MODEL,
        "--surrogate-b3-threshold", str(b3_threshold),
        "--run-name", run_name(profile, pavement, timesteps, seed),
        "--gen-backend", gen_backend,
        "--reward-profile", profile,
        "--gen-alpha-fallback", "0.0",
    ]
    if gen_model:
        cmd += ["--gen-model", gen_model]
    if not use_reranker:
        cmd.append("--gen-no-reranker")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward sensitivity training launcher")
    parser.add_argument("--profiles", default="all", help="Comma profiles or 'all'.")
    parser.add_argument("--pavement", default="flexible", choices=["flexible", "semi_rigid", "both"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--gen-backend", default="deepseek", choices=["deepseek", "chatfire", "siliconflow-qwen", "siliconflow-glm", "ollama", "ollama-llama"])
    parser.add_argument("--gen-model", default="deepseek-chat",
                        help="Structured-output model used for reward sensitivity.")
    parser.add_argument("--surrogate-b3-threshold", type=float, default=0.7,
                        help="Training-time B3 escalation threshold; use 1.0 for stricter but slower training.")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Enable LLM RAG reranker during training. Default off for speed/cost.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    profiles = REWARD_PROFILES if args.profiles == "all" else [x.strip() for x in args.profiles.split(",") if x.strip()]
    pavements = ["flexible", "semi_rigid"] if args.pavement == "both" else [args.pavement]
    seeds = parse_seeds(args.seeds)

    queue = []
    for profile in profiles:
        if profile not in REWARD_PROFILES:
            raise SystemExit(f"Unknown reward profile: {profile}")
        for pavement in pavements:
            for seed in seeds:
                name = run_name(profile, pavement, args.timesteps, seed)
                if not args.force and is_complete(name):
                    print(f"[skip] {name}")
                    continue
                queue.append((profile, pavement, seed, name))

    print(f"Reward sensitivity queue: {len(queue)} runs")
    for i, (profile, pavement, seed, name) in enumerate(queue, 1):
        cmd = build_command(profile, pavement, seed, args.timesteps, args.gen_backend, args.gen_model, args.surrogate_b3_threshold, args.use_reranker)
        print(f"[{i}/{len(queue)}] {name}")
        print("  " + " ".join(f'\"{c}\"' if " " in c else c for c in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()






