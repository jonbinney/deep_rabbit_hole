import argparse
import multiprocessing as mp
from pathlib import Path

from v2 import benchmarks, load_config_and_setup_run, self_play, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quoridor agent")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("-r", "--runs-dir", type=str, default=None, help="Directory for runs")
    # TODO: implement this
    # parser.add_argument("-c", "--continue", dest="continue_run", action="store_true", help="Continue an existing run")
    # parser.add_argument(
    #     "-o", "--overrides", nargs="*", help="Configuration overrides (e.g., run_id=my_run alphazero.mcts_n=250)"
    # )

    args = parser.parse_args()

    runs_dir = args.runs_dir if args.runs_dir is not None else str(Path(__file__).parent.parent)

    config = load_config_and_setup_run(args.config_file, runs_dir)
    mp.set_start_method("spawn", force=True)

    processes = []

    p = mp.Process(target=train, args=[config])
    p.start()
    processes.append(p)

    ps = benchmarks.create_benchmark_processes(config)
    [p.start() for p in ps]
    processes.extend(ps)

    for i in range(config.self_play.num_workers):
        p = mp.Process(target=self_play, args=[config])
        p.start()
        processes.append(p)

    for worker in processes:
        worker.join()
