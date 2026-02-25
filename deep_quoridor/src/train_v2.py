import argparse
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from v2 import benchmarks, load_config_and_setup_run, self_play, train
from v2.common import ShutdownSignal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quoridor agent")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "-r", "--runs-dir", type=str, default=None, help="Directory for runs"
    )
    # TODO: implement this
    # parser.add_argument("-c", "--continue", dest="continue_run", action="store_true", help="Continue an existing run")
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Configuration overrides (e.g., run_id=my_run alphazero.mcts_n=250)",
    )
    parser.add_argument(
        "--rust-selfplay",
        nargs="?",
        const="auto",
        default=None,
        help="Use Rust self-play binary. Optionally provide path; default: rust/target/release/selfplay",
    )

    args = parser.parse_args()

    runs_dir = (
        args.runs_dir
        if args.runs_dir is not None
        else str(Path(__file__).parent.parent)
    )

    # Resolve Rust self-play binary path
    rust_binary = None
    if args.rust_selfplay is not None:
        if args.rust_selfplay == "auto":
            rust_binary = str(
                Path(__file__).parent.parent
                / "rust"
                / "target"
                / "release"
                / "selfplay"
            )
        else:
            rust_binary = args.rust_selfplay
        if not Path(rust_binary).exists():
            print(f"ERROR: Rust self-play binary not found at {rust_binary}")
            print(
                "Build it with: cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay"
            )
            exit(1)

    # Apply overrides: force save_onnx when using Rust self-play
    overrides = args.overrides or []
    if rust_binary:
        overrides = list(overrides) + ["training.save_onnx=true"]

    config = load_config_and_setup_run(args.config_file, runs_dir, overrides=overrides)

    if rust_binary:
        config.self_play.rust_selfplay_binary = rust_binary

    mp.set_start_method("spawn", force=True)

    # Make sure we don't have the shutdown signal from a previous run
    ShutdownSignal.clear(config)

    train_process = mp.Process(target=train, args=[config])
    train_process.start()

    benchmark_processes = benchmarks.create_benchmark_processes(config)
    [p.start() for p in benchmark_processes]

    self_play_processes = []
    rust_subprocesses = []

    if rust_binary:
        # Spawn Rust self-play processes in continuous mode
        config_file_path = str(config.paths.config_file)
        for i in range(config.self_play.num_workers):
            cmd = [
                rust_binary,
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

    b_count_prev, sf_count_prev = -1, -1
    while True:
        b_count = sum([p.is_alive() for p in benchmark_processes])
        sf_count = sum([p.is_alive() for p in self_play_processes])
        sf_count += sum([p.poll() is None for p in rust_subprocesses])
        if b_count_prev != b_count or sf_count_prev != sf_count:
            print(
                f"Waiting for {b_count} benchmark processes and {sf_count} self_play processes"
            )
            b_count_prev, sf_count_prev = b_count, sf_count

        if (b_count + sf_count) == 0:
            break
        time.sleep(1)

    ShutdownSignal.clear(config)
