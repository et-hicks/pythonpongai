from __future__ import annotations

import argparse
import sys

import torch

from seeking.config import GridWorldConfig
from seeking.game.world import GridWorld
from seeking.rl.policy import PolicyNetwork
from seeking.rl.trainer import Trainer


def _has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device_arg: str) -> torch.device:
    arg = (device_arg or "cpu").lower()
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _has_mps():
            return torch.device("mps")
        return torch.device("cpu")
    if arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _has_mps():
            return torch.device("mps")
        return torch.device("cpu")
    if arg == "mps":
        if _has_mps():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def device_to_arg(device: torch.device) -> str:
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type


def describe_device(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(device.index or torch.cuda.current_device())
        except Exception:
            name = "CUDA"
        return f"CUDA ({name})"
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return "CPU"


def shape_label(shape: str) -> str:
    return "circle" if shape == "ball" else shape


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arcade + PyTorch playground.")
    parser.add_argument(
        "--mode",
        choices=["play", "train", "pong"],
        default="play",
        help="Run interactive play or headless training.",
    )
    shape_choices = ["ball", "square", "triangle"]
    parser.add_argument(
        "--pong-shape",
        choices=shape_choices,
        default="ball",
        help="Default projectile shape for Pong gameplay.",
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--obstacles", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Trainer learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if _has_mps() else "cpu")
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device identifier (e.g., cpu, cuda, mps, or auto).",
    )
    parser.add_argument(
        "--pong-train",
        action="store_true",
        help="Train the Pong RL agent instead of launching the playable window.",
    )
    parser.add_argument(
        "--pong-train-shape",
        choices=shape_choices,
        default="ball",
        help="Shape used for single-agent Pong training checkpoints.",
    )
    parser.add_argument(
        "--pong-green-shape",
        choices=shape_choices,
        default="ball",
        help="Shape associated with the green (left) Pong policy checkpoints.",
    )
    parser.add_argument(
        "--pong-purple-shape",
        choices=shape_choices,
        default="ball",
        help="Shape associated with the purple (right) Pong policy checkpoints.",
    )
    parser.add_argument(
        "--pong-competition",
        action="store_true",
        help="Train two Pong agents, run a battle, and promote the winner.",
    )
    parser.add_argument(
        "--pong-competition-shape",
        choices=shape_choices,
        default="ball",
        help="Projectile shape used during `--pong-competition` training.",
    )
    parser.add_argument(
        "--pong-selfplay",
        action="store_true",
        help="Launch the Pong window with both paddles training against each other.",
    )
    parser.add_argument(
        "--pong-selfplay-checkpoint",
        type=str,
        default="pong_selfplay_state.pt",
        help="Path used to persist self-play training state between runs.",
    )
    parser.add_argument(
        "--pong-checkpoint",
        type=str,
        default=None,
        help="Optional path to save the trained Pong policy (state_dict).",
    )
    parser.add_argument(
        "--pong-green-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for the green (left) Pong policy.",
    )
    parser.add_argument(
        "--pong-purple-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for the purple (right) Pong policy.",
    )
    parser.add_argument(
        "--pong-demo",
        type=str,
        default=None,
        help="Path to a trained Pong policy checkpoint to visualize in the UI.",
    )
    parser.add_argument(
        "--pong-demo-sample",
        action="store_true",
        help="Sample actions (instead of greedy) when running the Pong demo.",
    )
    parser.add_argument(
        "--pong-battle-sample",
        action="store_true",
        help="Sample actions (instead of greedy) when running the Pong battle UI.",
    )
    parser.add_argument(
        "--pong-run-shape",
        choices=shape_choices,
        default="ball",
        help="Projectile shape used when demoing or battling trained agents.",
    )
    parser.add_argument(
        "--pong-max-steps",
        type=int,
        default=600,
        help="Maximum steps per Pong training episode.",
    )
    parser.add_argument(
        "--pong-entropy-coef",
        type=float,
        default=0.02,
        help="Entropy regularization coefficient for Pong policy training.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable Arcade window.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def build_env(args: argparse.Namespace, headless: bool) -> GridWorld:
    config = GridWorldConfig(
        width=args.width,
        height=args.height,
        num_obstacles=args.obstacles,
        max_steps=args.max_steps,
        obstacle_seed=args.seed,
        headless=headless,
    )
    return GridWorld(config)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    resolved_device = resolve_device(args.device)
    device_string = device_to_arg(resolved_device)
    device_label = describe_device(resolved_device)
    if args.pong_checkpoint is None:
        args.pong_checkpoint = f"runs/policy_{shape_label(args.pong_train_shape)}.pt"
    if args.pong_green_checkpoint is None:
        args.pong_green_checkpoint = f"runs/green_{shape_label(args.pong_green_shape)}.pt"
    if args.pong_purple_checkpoint is None:
        args.pong_purple_checkpoint = f"runs/purple_{shape_label(args.pong_purple_shape)}.pt"
    if args.mode == "train":
        env = build_env(args, headless=True)
        policy = PolicyNetwork(env.observation_space_size, env.action_space_size)
        trainer = Trainer(env, policy, lr=args.lr, gamma=args.gamma, device=device_string)
        print(f"[Trainer] Using {device_label}")
        trainer.train(episodes=args.episodes)
    elif args.mode == "pong":
        toggles = [
            args.pong_selfplay,
            args.pong_train,
            args.pong_demo is not None,
            args.pong_competition,
        ]
        if sum(bool(t) for t in toggles) > 1:
            print(
                "Choose only one of --pong-selfplay, --pong-train, --pong-demo, or --pong-competition.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.pong_selfplay:
            from seeking.game.pong_selfplay import run_pong_selfplay

            run_pong_selfplay(
                device=resolved_device,
                lr=args.lr,
                gamma=args.gamma,
                checkpoint_path=args.pong_selfplay_checkpoint,
            )
        elif args.pong_train:
            from seeking.rl.pong_trainer import PongReinforceTrainer

            trainer = PongReinforceTrainer(
                device=resolved_device,
                lr=args.lr,
                gamma=args.gamma,
                max_steps=args.pong_max_steps,
                entropy_coef=args.pong_entropy_coef,
                seed=args.seed,
                projectile_shape=args.pong_train_shape,
            )
            trainer.train(episodes=args.episodes, checkpoint_path=args.pong_checkpoint)
            if args.pong_demo:
                demo_path = args.pong_demo
                from seeking.game.pong_policy_demo import run_pong_policy_demo

                run_pong_policy_demo(
                    checkpoint_path=demo_path,
                    device=resolved_device,
                    sample_actions=args.pong_demo_sample,
                    projectile_shape=args.pong_run_shape,
                )
        elif args.pong_competition:
            from seeking.game.pong_battle import run_pong_battle
            from seeking.rl.pong_dual_trainer import PongDualTrainer

            dual_trainer = PongDualTrainer(
                device=resolved_device,
                green_checkpoint=args.pong_green_checkpoint,
                purple_checkpoint=args.pong_purple_checkpoint,
                lr=args.lr,
                gamma=args.gamma,
                entropy_coef=args.pong_entropy_coef,
                seed=args.seed,
                projectile_shape=args.pong_competition_shape,
            )
            dual_trainer.train(episodes=args.episodes)
            dual_trainer.save()
            winner, left_score, right_score = run_pong_battle(
                green_checkpoint=args.pong_green_checkpoint,
                purple_checkpoint=args.pong_purple_checkpoint,
                device=resolved_device,
                sample_actions=args.pong_battle_sample,
                projectile_shape=args.pong_run_shape,
            )
            if winner:
                dual_trainer.promote_winner("green" if winner == "green" else "purple")
                print(
                    f"[Pong Competition] Winner: {winner.upper()} "
                    f"(score {left_score} - {right_score}). Both checkpoints updated."
                )
            else:
                print(
                    f"[Pong Competition] Battle ended in a tie ({left_score} - {right_score}). "
                    "Checkpoints left unchanged."
                )
        elif args.pong_demo:
            from seeking.game.pong_policy_demo import run_pong_policy_demo

            run_pong_policy_demo(
                checkpoint_path=args.pong_demo,
                device=resolved_device,
                sample_actions=args.pong_demo_sample,
                projectile_shape=args.pong_run_shape,
            )
        else:
            from seeking.game.pong import run_pong

            run_pong(
                device=resolved_device,
                green_checkpoint=args.pong_green_checkpoint,
                purple_checkpoint=args.pong_purple_checkpoint,
                default_shape=args.pong_shape,
                green_shape=args.pong_green_shape,
                purple_shape=args.pong_purple_shape,
            )
    else:
        if args.headless:
            print("Interactive mode requires Arcade window. Remove --headless.", file=sys.stderr)
            raise SystemExit(1)
        env = build_env(args, headless=False)
        from seeking.game.arcade_runner import run_interactive

        run_interactive(env)


if __name__ == "__main__":
    main()
