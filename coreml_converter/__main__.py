import argparse
import logging

import torch

from . import runner


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        CoreML Converter is a wrapper for coremltools to simply convert PyTorch model to CoreML model.
        """
    )
    parser.add_argument(
        "-W",
        "--weight",
        dest="weight_dir",
        type=str,
        default="./weights",
        help="Directory of model weight.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    runtime_conf = runner.RuntimeConfig(
        device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        weight_dir=args.weight_dir,
    )

    runner.run(runtime_config=runtime_conf)


logging.basicConfig()

if __name__ == "__main__":
    main()
