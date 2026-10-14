"""Command-line entry points for reproducible project demos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from sim2skill.dataset import ENV_VERSION, episode_splits, jsonable
from sim2skill.env import PickPlaceEnv
from sim2skill.expert import ScriptedExpert
from sim2skill.residual import ResidualPolicy, train_residual_policy
from sim2skill.rollout import collect_demonstrations, run_expert_episode
from sim2skill.safety import ActionShield
from sim2skill.wipe_env import WipeEnv
from sim2skill.wipe_expert import WipeExpert
from sim2skill.wipe_rollout import compare_wipe, evaluate_wipe, run_wipe_episode


def _pick_summary(episodes: int, seed: int) -> dict[str, float | int | dict[str, int]]:
    env = PickPlaceEnv(seed=seed)
    results = [run_expert_episode(env, seed + index).summary for index in range(episodes)]
    failures: dict[str, int] = {}
    for result in results:
        failures[result.failure] = failures.get(result.failure, 0) + 1
    return {
        "task": "pick",
        "episodes": episodes,
        "success_rate": sum(result.success for result in results) / episodes,
        "mean_steps": float(np.mean([result.steps for result in results])),
        "mean_final_distance": float(np.mean([result.final_distance for result in results])),
        "safety_interventions": sum(result.safety_interventions for result in results),
        "failures": failures,
    }


def _cmd_smoke(args: argparse.Namespace) -> None:
    if args.task == "wipe":
        print(json.dumps(evaluate_wipe(1, args.seed, "oracle", pose_noise=0.0), indent=2))
        return
    print(json.dumps(_pick_summary(episodes=1, seed=args.seed), indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    if args.task == "wipe":
        residual = ResidualPolicy.load(Path(args.residual)) if args.residual else None
        print(
            json.dumps(
                evaluate_wipe(
                    args.episodes,
                    args.seed,
                    args.policy,
                    args.pose_noise,
                    residual=residual,
                ),
                indent=2,
            )
        )
        return
    print(json.dumps(_pick_summary(episodes=args.episodes, seed=args.seed), indent=2))


def _cmd_collect(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    splits = episode_splits(args.episodes, args.seed)
    if args.task == "wipe":
        env = WipeEnv(seed=args.seed, pose_noise=args.pose_noise)
        traces = [
            run_wipe_episode(env, args.seed + index, "oracle")
            for index in range(args.episodes)
        ]
        dataset = {
            "observations": np.concatenate([trace.observations for trace in traces]),
            "actions": np.concatenate([trace.actions for trace in traces]),
            "stages": np.concatenate([trace.stages for trace in traces]),
            "episode_ids": np.concatenate(
                [
                    np.full(len(trace.actions), index, dtype=np.int64)
                    for index, trace in enumerate(traces)
                ]
            ),
        }
        summaries = [trace.summary for trace in traces]
        params = {"pose_noise": args.pose_noise, "expert": "oracle"}
    else:
        dataset, summaries = collect_demonstrations(args.episodes, args.seed)
        params = {"max_steps": 800}
    np.savez_compressed(output, **dataset)
    metadata = {
        "env_version": ENV_VERSION,
        "task": args.task,
        "seed": args.seed,
        "episodes": args.episodes,
        "transitions": int(len(dataset["actions"])),
        "split": splits,
        "split_rule": "episode-level permutation; frames from one episode never cross splits",
        "task_params": params,
        "episode_summaries": jsonable(summaries),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {metadata['transitions']} transitions to {output}")


def _cmd_demo(args: argparse.Namespace) -> None:
    if args.task == "wipe":
        env: PickPlaceEnv | WipeEnv = WipeEnv(seed=args.seed, pose_noise=args.pose_noise)
        expert_wipe = WipeExpert(use_true_pose=args.policy == "oracle")
        env.reset(seed=args.seed)
        expert_wipe.reset(env)
        shield = ActionShield(
            max_delta=0.028,
            workspace_low=np.array([-0.26, -0.20, 0.095]),
            workspace_high=np.array([0.26, 0.20, 0.31]),
        )
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running() and env.steps < env.max_steps and not expert_wipe.done:
                action = shield.filter(expert_wipe.action(env), env.eef_position).action
                result = env.step(action)
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip)
                if result.truncated:
                    break
        print(json.dumps({"success": bool(result.info["success"]), "steps": env.steps}, indent=2))
        return

    env = PickPlaceEnv(seed=args.seed)
    expert = ScriptedExpert()
    shield = ActionShield()
    env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and env.steps < env.max_steps:
            action = shield.filter(expert.action(env), env.eef_position).action
            result = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.frame_skip)
            if result.terminated or result.truncated:
                time.sleep(1.0)
                break
    print(json.dumps({"success": bool(result.info["success"]), "steps": env.steps}, indent=2))


def _cmd_train_residual(args: argparse.Namespace) -> None:
    output = Path(args.output)
    policy, losses = train_residual_policy(args.episodes, args.seed, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "train_episodes": args.episodes,
                "final_loss": losses[-1] if losses else None,
                "loss_curve": losses,
                "bound": policy.bound,
            },
            indent=2,
        )
    )


def _cmd_compare(args: argparse.Namespace) -> None:
    residual_path = Path(args.residual)
    if args.train_episodes > 0 and not residual_path.exists():
        train_residual_policy(args.train_episodes, args.seed, residual_path)
    rows = compare_wipe(args.episodes, args.seed, residual_path)
    print(json.dumps({"env_version": ENV_VERSION, "rows": rows}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim2skill",
        description="Simulation-first embodied manipulation baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run one headless expert episode")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--task", choices=("pick", "wipe"), default="pick")
    smoke.set_defaults(func=_cmd_smoke)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a baseline policy")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--task", choices=("pick", "wipe"), default="pick")
    evaluate.add_argument(
        "--policy",
        choices=("nominal", "oracle", "residual", "structured"),
        default="nominal",
    )
    evaluate.add_argument("--pose-noise", type=float, default=0.0)
    evaluate.add_argument("--residual", default="")
    evaluate.set_defaults(func=_cmd_evaluate)

    collect = subparsers.add_parser("collect", help="collect expert demonstrations")
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--output", default="artifacts/demos.npz")
    collect.add_argument("--task", choices=("pick", "wipe"), default="pick")
    collect.add_argument("--pose-noise", type=float, default=0.0)
    collect.set_defaults(func=_cmd_collect)

    demo = subparsers.add_parser("demo", help="open an interactive MuJoCo viewer")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--task", choices=("pick", "wipe"), default="pick")
    demo.add_argument("--policy", choices=("nominal", "oracle"), default="oracle")
    demo.add_argument("--pose-noise", type=float, default=0.0)
    demo.set_defaults(func=_cmd_demo)

    train = subparsers.add_parser("train-residual", help="train a bounded wipe residual")
    train.add_argument("--episodes", type=int, default=16)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", default="artifacts/residual.npz")
    train.set_defaults(func=_cmd_train_residual)

    compare = subparsers.add_parser("compare", help="compare nominal, oracle, and residual wipe")
    compare.add_argument("--episodes", type=int, default=8)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--train-episodes", type=int, default=16)
    compare.add_argument("--residual", default="artifacts/residual.npz")
    compare.set_defaults(func=_cmd_compare)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
