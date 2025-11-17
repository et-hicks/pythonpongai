from __future__ import annotations

import argparse
import sys

import torch

from seeking.config import GridWorldConfig
from seeking.game.world import GridWorld
from seeking.rl.policy import PolicyNetwork
from seeking.rl.trainer import Trainer


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arcade + PyTorch playground.")
    parser.add_argument(
        "--mode",
        choices=["play", "train", "pong"],
        default="play",
        help="Run interactive play or headless training.",
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--obstacles", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Trainer learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier.",
    )
    parser.add_argument(
        "--pong-train",
        action="store_true",
        help="Train the Pong RL agent instead of launching the playable window.",
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
    if args.mode == "train":
        env = build_env(args, headless=True)
        policy = PolicyNetwork(env.observation_space_size, env.action_space_size)
        trainer = Trainer(env, policy, lr=args.lr, gamma=args.gamma, device=args.device)
        trainer.train(episodes=args.episodes)
    elif args.mode == "pong":
        if args.pong_selfplay and (args.pong_train or args.pong_demo):
            print(
                "Choose either --pong-selfplay, --pong-train, or --pong-demo (only one at a time).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.pong_selfplay:
            from seeking.game.pong_selfplay import run_pong_selfplay

            device = torch.device(args.device)
            run_pong_selfplay(
                device=device,
                lr=args.lr,
                gamma=args.gamma,
                checkpoint_path=args.pong_selfplay_checkpoint,
            )
        elif args.pong_train:
            from seeking.rl.pong_trainer import PongReinforceTrainer

            device = torch.device(args.device)
            trainer = PongReinforceTrainer(
                device=device,
                lr=args.lr,
                gamma=args.gamma,
                max_steps=args.pong_max_steps,
                entropy_coef=args.pong_entropy_coef,
                seed=args.seed,
            )
            trainer.train(episodes=args.episodes, checkpoint_path=args.pong_checkpoint)
            if args.pong_demo:
                demo_path = args.pong_demo
                from seeking.game.pong_policy_demo import run_pong_policy_demo

                run_pong_policy_demo(
                    checkpoint_path=demo_path,
                    device=device,
                    sample_actions=args.pong_demo_sample,
                )
        elif args.pong_demo:
            from seeking.game.pong_policy_demo import run_pong_policy_demo

            device = torch.device(args.device)
            run_pong_policy_demo(
                checkpoint_path=args.pong_demo,
                device=device,
                sample_actions=args.pong_demo_sample,
            )
        else:
            from seeking.game.pong import run_pong

            run_pong()
    else:
        if args.headless:
            print("Interactive mode requires Arcade window. Remove --headless.", file=sys.stderr)
            raise SystemExit(1)
        env = build_env(args, headless=False)
        from seeking.game.arcade_runner import run_interactive

        run_interactive(env)


if __name__ == "__main__":
    main()
