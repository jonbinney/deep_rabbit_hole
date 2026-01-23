import multiprocessing as mp
import re
import time
from typing import Optional

import wandb
from config import (
    AgentEvolutionBenchmarkConfig,
    BenchmarkScheduleConfig,
    Config,
    DumbScoreBenchmarkConfig,
    TournamentBenchmarkConfig,
)
from pydantic_yaml import parse_yaml_file_as
from v2.common import create_alphazero

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


# TODO: move to common.py, split into 2 classes, one for time and another for model, that extend the
# same base class.  The base class will have an abstract "wait" method, and the static constructor.
# Also, figure out a better name for the classes (e.g. TimeTrigger and ModelTrigger ? or Scheduler?)
class RunFrequency:
    def __init__(self, every_s: Optional[int] = None, every_models: Optional[int] = None):
        assert every_s or every_models, "Either every_s or every_models need to be set"
        # only one
        self.every_s = every_s
        self.every_models = every_models
        self.next = None

    @classmethod
    def from_string(cls, s: str) -> "RunFrequency":
        match = re.match(r"^\s*(\d+)\s*(second|seconds|minute|minutes|hour|hours|model|models)\s*$", s.lower())
        if not match:
            raise ValueError(f"Invalid frequency string: {s!r}")

        value = int(match.group(1))
        if value <= 0:
            raise ValueError(f"Frequency must be a positive integer, got {value}")

        unit = match.group(2)
        if unit in ("model", "models"):
            return cls(every_models=value)

        seconds_per_unit = {
            "second": 1,
            "seconds": 1,
            "minute": 60,
            "minutes": 60,
            "hour": 3600,
            "hours": 3600,
        }
        return cls(every_s=value * seconds_per_unit[unit])

    def wait(self):
        if self.every_s:
            t = time.time()
            if self.next is None:
                self.next = t + self.every_s
                return

            if t < self.next:
                time.sleep(self.next - t)

            self.next = time.time() + self.every_s

        # TODO every_models


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
    freq = RunFrequency.from_string(benchmark.every)

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


# config = load_config_and_setup_run("deep_quoridor/experiments/B5W3/demo.yaml", "/Users/amarcu/code/deep_rabbit_hole")

# run_benchmark(config, config.benchmarks[0])
