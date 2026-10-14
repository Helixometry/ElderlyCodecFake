"""Evaluation script for ElderlyCodecFake."""

import argparse
import torch
from ecf.model import BONSAI
from ecf.utils import get_device


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate BONSAI model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test-data", type=str, required=True, help="Test data directory")

    args = parser.parse_args()

    device = get_device()
    model = BONSAI.from_pretrained(args.checkpoint).to(device)

    print(f"Loaded model from {args.checkpoint}")
    print(f"Evaluating on {args.test_data}")


if __name__ == "__main__":
    main()
