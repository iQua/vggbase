"""
Implementation of canceling multiples experiments with Slurm.

"""

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ids",
        nargs="+",
        default=[],
        help="The jobs with ids will be cancelled.",
    )

    args = parser.parse_args()

    jobs_id = [int(input_id) for input_id in args.ids]
    print(f"Cancelling job with id: {jobs_id}")

    for work_id in jobs_id:
        os.system(f"scancel {work_id}")
