import argparse
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

from v2 import benchmarks, check_ai_available, load_config_and_setup_run, run_ai_reporter, self_play, train
from v2.common import ShutdownSignal

# Prevents getting messages in the console every few lines telling you to install weave
os.environ["WANDB_DISABLE_WEAVE"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quoridor agent")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("-r", "--runs-dir", type=str, default=None, help="Directory for runs")
    # TODO: implement this
    # parser.add_argument("-c", "--continue", dest="continue_run", action="store_true", help="Continue an existing run")
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Configuration overrides (e.g., run_id=my_run self_play.program=rust)",
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir if args.runs_dir is not None else str(Path(__file__).parent.parent)

    config = load_config_and_setup_run(args.config_file, runs_dir, overrides=args.overrides)

    # Validate AI report prerequisites before spawning anything, so a misconfigured
    # run aborts early instead of failing silently inside a sibling process.
    if config.ai_report is not None:
        try:
            check_ai_available(config.ai_report.ai)
        except Exception as e:
            print(f"ERROR: {e}")
            exit(1)

    mp.set_start_method("spawn", force=True)

    # Make sure we don't have the shutdown signal from a previous run
    ShutdownSignal.clear(config)

    train_process = mp.Process(target=train, args=[config])
    train_process.start()

    benchmark_processes = benchmarks.create_benchmark_processes(config)
    [p.start() for p in benchmark_processes]

    ai_report_process = None
    if config.ai_report is not None:
        ai_report_process = mp.Process(target=run_ai_reporter, args=[config])
        ai_report_process.start()

    self_play_processes = []
    rust_subprocesses = []

    if config.self_play.program == "rust":
        # Spawn Rust self-play processes in continuous mode
        config_file_path = str(config.paths.config_file)
        for i in range(config.self_play.num_workers):
            cmd = [
                config.self_play.rust_selfplay_binary,
                "--config",
                config_file_path,
                "--output-dir",
                str(config.paths.replay_buffers_ready),
                "--continuous",
                "--latest-model-yaml",
                str(config.paths.latest_model_yaml),
                "--shutdown-file",
                str(ShutdownSignal.file_path(config)),
            ]
            proc = subprocess.Popen(cmd)
            rust_subprocesses.append(proc)
            print(f"Started Rust self-play process {proc.pid}")
    else:
        for i in range(config.self_play.num_workers):
            p = mp.Process(target=self_play, args=[config])
            p.start()
            self_play_processes.append(p)

    train_process.join()
    ShutdownSignal.signal(config)
    print("Shutting down!")

    b_count_prev, sf_count_prev, ai_count_prev = -1, -1, -1
    while True:
        b_count = sum([p.is_alive() for p in benchmark_processes])
        sf_count = sum([p.is_alive() for p in self_play_processes])
        sf_count += sum([p.poll() is None for p in rust_subprocesses])
        ai_count = 1 if ai_report_process is not None and ai_report_process.is_alive() else 0
        if b_count_prev != b_count or sf_count_prev != sf_count or ai_count_prev != ai_count:
            print(
                f"Waiting for {b_count} benchmark processes, {sf_count} self_play processes"
                f" and {ai_count} ai_report processes"
            )
            b_count_prev, sf_count_prev, ai_count_prev = b_count, sf_count, ai_count

        if (b_count + sf_count + ai_count) == 0:
            break
        time.sleep(1)

    ShutdownSignal.clear(config)
