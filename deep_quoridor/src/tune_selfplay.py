import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SELFPLAY_RE = re.compile(r"selfplay finished in ([\d.]+)")


def parse_selfplay_line(line):
    """Parse a self-play round completion line.

    Returns (elapsed_seconds, num_games) or None if the line doesn't match.
    """
    m = SELFPLAY_RE.search(line)
    if not m:
        return None

    elapsed = float(m.group(1))

    return elapsed


def run_single_benchmark(config_file, num_workers, parallel_games, duration, runs_dir, extra_overrides):
    run_id = f"bench-w{num_workers}-g{parallel_games}-{int(time.time())}"

    overrides = [
        f"run_id={run_id}",
        f"self_play.num_workers={num_workers}",
        f"self_play.parallel_games={parallel_games}",
        f"training.finish_after={duration}",
        "benchmarks=[]",
        "wandb=None",
    ] + extra_overrides

    cmd = [sys.executable, "train_v2.py", config_file, "-o"] + overrides
    if runs_dir:
        cmd.insert(3, "--runs-dir")
        cmd.insert(4, runs_dir)

    src_dir = Path(__file__).parent

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(src_dir),
    )
    durations = []
    all_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        all_lines.append(line)
        parsed = parse_selfplay_line(line)
        if parsed:
            print(f"  {line}")
            durations.append(parsed)

    proc.wait()

    # Cleanup run directory
    effective_runs_dir = runs_dir or str(src_dir.parent)
    run_dir = Path(effective_runs_dir) / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    if not durations:
        print("WARNING: couldn't find the duration of the self-play")
        print(all_lines)

    return compute_metrics(num_workers, parallel_games, durations)


def compute_metrics(num_workers, parallel_games, durations):
    if not durations:
        return {
            "num_workers": num_workers,
            "parallel_games": parallel_games,
            "total_rounds": 0,
            "total_games": 0,
            "avg_round_time": float("nan"),
            "avg_throughput": 0.0,
        }

    total_rounds = len(durations)
    total_worker_time = sum(durations)
    total_games = parallel_games * total_rounds
    avg_throughput = num_workers * total_games / total_worker_time

    return {
        "num_workers": num_workers,
        "parallel_games": parallel_games,
        "total_rounds": total_rounds,
        "total_games": total_games,
        "avg_round_time": total_worker_time / total_rounds,
        "avg_throughput": avg_throughput,
    }


def print_results_table(results):
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 80}")

    header = f"{'nw':>10} {'pg':>10} {'rounds':>10} {'tot games':>10} {'avg round t':>14} {'games/s':>14}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["avg_throughput"], reverse=True):
        print(
            f"{r['num_workers']:>10} "
            f"{r['parallel_games']:>10} "
            f"{r['total_rounds']:>10} "
            f"{r['total_games']:>10} "
            f"{r['avg_round_time']:>14.2f} "
            f"{r['avg_throughput']:>14.3f} "
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark self-play throughput across configurations")
    parser.add_argument("config_file", type=str, help="Base config YAML file")
    parser.add_argument("--workers", type=str, help="Comma-separated num_workers values")
    parser.add_argument("--games", type=str, help="Comma-separated parallel_games values")
    parser.add_argument("--duration", type=str, default="2 minutes", help="Duration per combo (default: '2 minutes')")
    parser.add_argument("--runs-dir", type=str, default=None, help="Directory for runs")
    parser.add_argument("-o", "--overrides", nargs="*", default=[], help="Additional config overrides for train_v2.py")
    args = parser.parse_args()

    workers_list = [int(x) for x in args.workers.split(",")]
    games_list = [int(x) for x in args.games.split(",")]
    combos = [(w, g) for w in workers_list for g in games_list]

    print(f"Benchmarking {len(combos)} configurations, {args.duration} each")
    print(f"  Workers: {workers_list}")
    print(f"  Parallel games: {games_list}")

    results = []
    for i, (num_workers, parallel_games) in enumerate(combos):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(combos)}] num_workers={num_workers}, parallel_games={parallel_games}")
        print(f"{'=' * 60}")

        result = run_single_benchmark(
            config_file=args.config_file,
            num_workers=num_workers,
            parallel_games=parallel_games,
            duration=args.duration,
            runs_dir=args.runs_dir,
            extra_overrides=args.overrides,
        )
        results.append(result)

    print_results_table(results)


if __name__ == "__main__":
    main()
