#!/bin/env python3

#!/usr/bin/env python3

import argparse
import pstats

def main():
    parser = argparse.ArgumentParser(
        description="View and print Python cProfile stats from a file."
    )
    parser.add_argument(
        "profile_file",
        help="The cProfile stats file to read (e.g., output.prof)"
    )
    parser.add_argument(
        "-n", "--num-lines",
        type=int,
        default=20,
        help="Number of lines to print (default: 20)"
    )
    parser.add_argument(
        "-s", "--sort",
        default="cumulative",
        choices=["cumulative", "time", "calls", "name", "filename", "module", "pcalls"],
        help="Sort order for stats (default: cumulative)"
    )

    args = parser.parse_args()

    try:
        stats = pstats.Stats(args.profile_file)
        stats.strip_dirs().sort_stats(args.sort).print_stats(args.num_lines)
    except FileNotFoundError:
        print(f"File '{args.profile_file}' not found.")
    except Exception as e:
        print(f"Error reading profile: {e}")

if __name__ == "__main__":
    main()


