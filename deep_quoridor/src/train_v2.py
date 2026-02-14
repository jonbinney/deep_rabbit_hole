import argparse
import multiprocessing as mp
import time
from pathlib import Path

from v2 import benchmarks, load_config_and_setup_run, self_play, train
from v2.common import ShutdownSignal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quoridor agent")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("-r", "--runs-dir", type=str, default=None, help="Directory for runs")
    # TODO: implement this
    # parser.add_argument("-c", "--continue", dest="continue_run", action="store_true", help="Continue an existing run")
    parser.add_argument(
        "-o", "--overrides", nargs="*", help="Configuration overrides (e.g., run_id=my_run alphazero.mcts_n=250)"
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir if args.runs_dir is not None else str(Path(__file__).parent.parent)

    config = load_config_and_setup_run(args.config_file, runs_dir, overrides=args.overrides)
    mp.set_start_method("spawn", force=True)

    # Make sure we don't have the shutdown signal from a previous run
    ShutdownSignal.clear(config)

    train_process = mp.Process(target=train, args=[config])
    train_process.start()

    benchmark_processes = benchmarks.create_benchmark_processes(config)
    [p.start() for p in benchmark_processes]

    self_play_processes = []
    for i in range(config.self_play.num_workers):
        p = mp.Process(target=self_play, args=[config])
        p.start()
        self_play_processes.append(p)

    train_process.join()
    ShutdownSignal.signal(config)
    print("Shutting down!")

    b_count_prev, sf_count_prev = -1, -1
    while True:
        b_count = sum([p.is_alive() for p in benchmark_processes])
        sf_count = sum([p.is_alive() for p in self_play_processes])
        if b_count_prev != b_count or sf_count_prev != sf_count:
            print(f"Waiting for {b_count} benchmark processes and {sf_count} self_play processes")
            b_count_prev, sf_count_prev = b_count, sf_count

        if (b_count + sf_count) == 0:
            break
        time.sleep(1)

    ShutdownSignal.clear(config)
