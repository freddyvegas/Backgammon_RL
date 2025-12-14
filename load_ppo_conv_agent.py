#!/usr/bin/env python3
"""Utility script to load a PPOConvAgent checkpoint with a chosen config."""

from argparse import ArgumentParser
from pathlib import Path
import sys

from ppo_conv_agent import PPOConvAgent, get_config


def parse_args():
    parser = ArgumentParser(description="Load a PPOConvAgent checkpoint.")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "model_size",
        choices=["small", "medium", "large"],
        help="Model configuration to instantiate before loading weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = args.checkpoint.expanduser()

    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    config = get_config(args.model_size)
    agent = PPOConvAgent(config=config, device="cpu")

    agent.load(str(ckpt_path), map_location="cpu", load_optimizer=False)
    agent.set_eval_mode(True)

    print(f"\nLoaded PPOConvAgent from {ckpt_path}")
    print(f"  Model size: {args.model_size}")
    print(f"  Device: {agent.device}")


if __name__ == "__main__":
    main()
