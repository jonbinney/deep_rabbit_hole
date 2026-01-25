import multiprocessing as mp
import os
import time
from abc import abstractmethod

import torch
import wandb
from config import (
    AgentEvolutionBenchmarkConfig,
    BenchmarkScheduleConfig,
    Config,
    DumbScoreBenchmarkConfig,
    TournamentBenchmarkConfig,
    load_config_and_setup_run,
)
from metrics import Metrics
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


class BenchmarkJob:
    @classmethod
    def from_job_config(cls, config: Config, job_config):
        if isinstance(job_config, TournamentBenchmarkConfig):
            return TournamentBenchmarkJob(config, job_config)

        if isinstance(job_config, DumbScoreBenchmarkConfig):
            return DumbScoreBenchmarkJob(config, job_config)

        if isinstance(job_config, AgentEvolutionBenchmarkConfig):
            return AgentEvolutionBenchmarkJob(config, job_config)

        raise ValueError(f"Unknown job config type: {job_config}")

    @abstractmethod
    def run(self, model_filename: str):
        pass


class TournamentBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: TournamentBenchmarkConfig):
        self.job_config = job_config

    def run(self, model_filename: str):
        print(f"Tournamet score:, prefix: {self.job_config}, filename {model_filename}")

        pass


class DumbScoreBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: DumbScoreBenchmarkConfig):
        self.config = config
        self.job_config = job_config
        self.metrics = Metrics(config.quoridor.board_size, config.quoridor.max_walls)  # antyhing else important?

    def run(self, model_filename: str):
        # We create the agent with temperature 0 for Dumb Score, since we want to see it in its best behavior.
        # In real playing, the temperature should be 0 or have dropped to 0 before getting to a terminal situation
        # like the ones in dumb score.
        agent = create_alphazero(
            self.config, self.job_config.alphazero, overrides={"model_filename": model_filename, "temperature": 0}
        )

        score = self.metrics.dumb_score(agent, verbose=False)
        print(f"Dumb score: {score}, prefix: {self.job_config.prefix}, {agent.params.model_filename}")

        # TODO log to wandb

        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AgentEvolutionBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: AgentEvolutionBenchmarkConfig):
        pass

    def run(self, model_filename: str):
        pass


def run_benchmark(config: Config, benchmark: BenchmarkScheduleConfig):
    freq = JobTrigger.from_string(config, benchmark.every)
    LatestModel.wait_for_creation(config)

    jobs = [BenchmarkJob.from_job_config(config, job_config) for job_config in benchmark.jobs]

    while True:
        freq.wait()
        # We get the model filename here so that all the jobs run with the same model
        model_filename = LatestModel.load(config).filename

        print(f"=== ({os.getpid()} running benchmark with {model_filename} ===")
        for job in jobs:
            job.run(model_filename)


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
