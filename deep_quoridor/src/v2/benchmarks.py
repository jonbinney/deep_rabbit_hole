import multiprocessing as mp
from abc import abstractmethod

import torch
import wandb
from metrics import Metrics
from v2.common import JobTrigger, MockWandb, create_alphazero
from v2.config import (
    AgentEvolutionBenchmarkConfig,
    BenchmarkScheduleConfig,
    Config,
    DumbScoreBenchmarkConfig,
    TournamentBenchmarkConfig,
)
from v2.yaml_models import LatestModel


class BenchmarkJob:
    @classmethod
    def from_job_config(cls, config: Config, job_config, wandb_run):
        if isinstance(job_config, TournamentBenchmarkConfig):
            return TournamentBenchmarkJob(config, job_config, wandb_run)

        if isinstance(job_config, DumbScoreBenchmarkConfig):
            return DumbScoreBenchmarkJob(config, job_config, wandb_run)

        if isinstance(job_config, AgentEvolutionBenchmarkConfig):
            return AgentEvolutionBenchmarkJob(config, job_config, wandb_run)

        raise ValueError(f"Unknown job config type: {job_config}")

    @abstractmethod
    def run(self, latest: LatestModel):
        pass


class TournamentBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: TournamentBenchmarkConfig, wandb_run):
        self.config = config
        self.job_config = job_config
        self.wandb_run = wandb_run
        self.metrics = Metrics(
            config.quoridor.board_size,
            config.quoridor.max_walls,
            benchmarks=job_config.opponents,
            benchmarks_t=job_config.times,
            max_steps=config.quoridor.max_steps,
        )

    def run(self, latest: LatestModel):
        agent = create_alphazero(self.config, self.job_config.alphazero, overrides={"model_filename": latest.filename})
        (
            _,
            relative_elo,
            win_perc,
            p1_stats,
            p2_stats,
            absolute_elo,
        ) = self.metrics.tournament(agent)
        prefix = self.job_config.prefix
        if prefix != "":
            prefix = prefix + "_"

        metrics = {
            f"{prefix}relative_elo": relative_elo,
            f"{prefix}win_perc": win_perc,
            f"{prefix}absolute_elo": absolute_elo,
            "Model version": latest.version,
        }

        metrics.update(self.metrics.metrics_from_stats(prefix, p1_stats, p2_stats))

        self.wandb_run.log(metrics)


class DumbScoreBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: DumbScoreBenchmarkConfig, wandb_run):
        self.config = config
        self.job_config = job_config
        self.wandb_run = wandb_run
        self.metrics = Metrics(config.quoridor.board_size, config.quoridor.max_walls)

    def run(self, latest: LatestModel):
        # We create the agent with temperature 0 for Dumb Score, since we want to see it in its best behavior.
        # In real playing, the temperature should be 0 or have dropped to 0 before getting to a terminal situation
        # like the ones in dumb score.
        agent = create_alphazero(
            self.config, self.job_config.alphazero, overrides={"model_filename": latest.filename, "temperature": 0}
        )
        score = self.metrics.dumb_score(agent, verbose=False)

        prefix = self.job_config.prefix
        if prefix != "":
            prefix = prefix + "_"
        self.wandb_run.log({f"{prefix}dumb_score": score, "Model version": latest.version})

        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AgentEvolutionBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: AgentEvolutionBenchmarkConfig, wandb_run):
        self.config = config
        self.job_config = job_config
        self.wandb_run = wandb_run

    def run(self, latest: LatestModel):
        pass


def run_benchmark(config: Config, benchmark: BenchmarkScheduleConfig):
    freq = JobTrigger.from_string(config, benchmark.every)
    LatestModel.wait_for_creation(config)
    if config.wandb:
        idx = config.benchmarks.index(benchmark)
        run_id = f"{config.run_id}-benchmark-{idx}"
        wandb_run = wandb.init(
            project=config.wandb.project,
            job_type="benchmark",
            group=config.run_id,
            name=run_id,
            id=run_id,
            resume="allow",
        )
        wandb.define_metric("Model version", hidden=True)
        wandb.define_metric("*", "Model version")
    else:
        wandb_run = MockWandb()

    jobs = [BenchmarkJob.from_job_config(config, job_config, wandb_run) for job_config in benchmark.jobs]

    while True:
        freq.wait()
        # We get the model filename here so that all the jobs run with the same model
        latest = LatestModel.load(config)

        for job in jobs:
            job.run(latest)

        wandb_run.log({"Model version": latest.version}, commit=True)


def create_benchmark_processes(config: Config) -> list[mp.Process]:
    ps = []
    for benchmark in config.benchmarks:
        p = mp.Process(target=run_benchmark, args=[config, benchmark])
        ps.append(p)

    return ps
