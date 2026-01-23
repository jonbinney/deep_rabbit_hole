import multiprocessing as mp
import time

import wandb
from config import (
    AgentEvolutionBenchmarkConfig,
    BenchmarkScheduleConfig,
    Config,
    DumbScoreBenchmarkConfig,
    TournamentBenchmarkConfig,
    load_config_and_setup_run,
)
from pydantic_yaml import parse_yaml_file_as
from v2.common import JobTrigger, LatestModel, create_alphazero

# from self_play import LatestModel


def benchmarks(config: Config):
    time.sleep(10)  # to do, just wait until it's available or a trigger

    if config.wandb:
        run_id = f"{config.run_id}-training"
        wandb_run = wandb.init(
            project=config.wandb.project,
            job_type="benchmark",
            group=config.run_id,
            name=run_id,
            id=run_id,
            resume="allow",
        )
    else:
        wandb_run = None

    while True:
        latest = parse_yaml_file_as(LatestModel, config.paths.latest_model_yaml)

        az_params_override = override_subargs(
            azparams_str, {"mcts_n": 0, "model_filename": latest.filename, "training_mode": False}
        )
        params = f"alphazero:{az_params_override}"

        players: list[str] = [
            "greedy",
            "greedy:p_random=0.1,nick=greedy-01",
            "greedy:p_random=0.3,nick=greedy-03",
            "random",
            "simple:branching_factor=8,nick=simple-bf8",
            "simple:branching_factor=16,nick=simple-bf16",
        ]
        m = Metrics(5, 3, players, 10, 30, 1)
        print("METRICS - starting computation")
        (
            _,
            _,
            relative_elo,
            win_perc,
            p1_stats,
            p2_stats,
            absolute_elo,
            dumb_score,
        ) = m.compute(params)

        print(f"METRICS {latest.version}: {relative_elo=}, {win_perc=}, {dumb_score=}")
        wandb_run.log(
            {"win_perc": win_perc, "relative_elo": relative_elo, "dumb_score": dumb_score},
            step=latest.version,
            commit=True,
        )
        time.sleep(60)


def run_tournament_benchmark(config: Config, job: TournamentBenchmarkConfig) -> None:
    print(f"Running tournament benchmark: {job.prefix}")


def run_dumb_score_benchmark(config: Config, job: DumbScoreBenchmarkConfig) -> None:
    print(f"Running dumb score benchmark: {job.prefix}")


def run_agent_evolution_benchmark(config: Config, job: AgentEvolutionBenchmarkConfig) -> None:
    print(f"Running agent evolution benchmark: {job.prefix}")


def run_benchmark_job(config: Config, job) -> None:
    # pass filename, needs to come from args
    agent = create_alphazero(config, job.alphazero, training_mode=False)

    if isinstance(job, TournamentBenchmarkConfig):
        run_tournament_benchmark(config, job)
        return

    if isinstance(job, DumbScoreBenchmarkConfig):
        run_dumb_score_benchmark(config, job)
        return

    if isinstance(job, AgentEvolutionBenchmarkConfig):
        run_agent_evolution_benchmark(config, job)
        return

    raise ValueError(f"Unknown benchmark job type: {type(job).__name__}")


def run_benchmark(config: Config, benchmark: BenchmarkScheduleConfig):
    freq = JobTrigger.from_string(config, benchmark.every)

    while True:
        freq.wait()
        for job in benchmark.jobs:
            run_benchmark_job(config, job)


def create_benchmark_processes(config: Config) -> list[mp.Process]:
    ps = []
    for benchmark in config.benchmarks:
        p = mp.Process(target=run_benchmark, args=[config, benchmark])
        ps.append(p)

    return ps


if __name__ == "__main__":
    config = load_config_and_setup_run(
        "deep_quoridor/experiments/B5W3/demo.yaml", "/Users/amarcu/code/deep_rabbit_hole"
    )

    # run_benchmark(config, config.benchmarks[0])
    freq = JobTrigger.from_string(config, "10 models")
    while True:
        freq.wait()
        print(LatestModel.load(config).version)
