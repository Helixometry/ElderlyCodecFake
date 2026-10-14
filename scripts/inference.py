"""Inference script for ElderlyCodecFake."""

import argparse
import torch
from pathlib import Path

from ecf.model import BONSAI
from ecf.utils import get_device


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference with BONSAI")
    parser.add_argument("--audio", type=str, required=True, help="Audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint")

    args = parser.parse_args()

    device = get_device()
    model = BONSAI.from_pretrained(args.checkpoint).to(device)

    print(f"\nElderlyCodecFake Inference Results")
    print("=" * 60)
    print(f"Audio: {Path(args.audio).name}")
    print(f"\n(Placeholder inference)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
