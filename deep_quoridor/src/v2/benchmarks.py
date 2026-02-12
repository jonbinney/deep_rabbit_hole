import multiprocessing as mp
from abc import abstractmethod

import torch
import wandb
from agent_evolution_tournament import AgentEvolutionTournament, AgentEvolutionTournamentParams
from metrics import Metrics
from utils import Timer
from v2.common import (
    JobTrigger,
    MockWandb,
    ShutdownSignal,
    alphazero_encoded_name_from_config,
    create_alphazero,
    upload_model,
)
from v2.config import (
    AgentEvolutionBenchmarkConfig,
    BenchmarkScheduleConfig,
    Config,
    DumbScoreBenchmarkConfig,
    TournamentBenchmarkConfig,
)
from v2.yaml_models import LatestModel


class BenchmarkJob:
    def __init__(self, config: Config, wandb_run, prefix: str):
        self.config = config
        self.wandb_run = wandb_run
        self.prefix = "" if prefix == "" else prefix + "_"
        self.metrics_max = {}
        self.metrics_min = {}

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
    def _run(self, latest: LatestModel):
        pass

    def run(self, latest: LatestModel):
        print(f"Running {self.__class__.__name__} with prefix '{self.prefix}' on model version {latest.version}")
        self._run(latest)
        print(
            f"Finished running {self.__class__.__name__} with prefix '{self.prefix}' on model version {latest.version}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def upload_model_if_needed(self, metrics, model: LatestModel, model_id: str):
        if not (self.config.wandb and self.config.wandb.upload_model):
            return

        tags = []
        for metric_name in self.config.wandb.upload_model.when_max:
            if metric_name not in metrics:
                continue

            if metric_name not in self.metrics_max or metrics[metric_name] > self.metrics_max[metric_name]:
                self.metrics_max[metric_name] = metrics[metric_name]
                tags.append(f"max-{metric_name}-{self.config.run_id}")

        for metric_name in self.config.wandb.upload_model.when_min:
            if metric_name not in metrics:
                continue

            if metric_name not in self.metrics_min or metrics[metric_name] < self.metrics_min[metric_name]:
                self.metrics_min[metric_name] = metrics[metric_name]
                tags.append(f"min-{metric_name}-{self.config.run_id}")

        if tags:
            tags.append(f"m{model.version}-{self.config.run_id}")
            upload_model(self.wandb_run, self.config, model, model_id, tags)


class TournamentBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: TournamentBenchmarkConfig, wandb_run):
        super().__init__(config, wandb_run, job_config.prefix)
        self.job_config = job_config
        self.metrics = Metrics(
            config.quoridor.board_size,
            config.quoridor.max_walls,
            benchmarks=job_config.opponents,
            benchmarks_t=job_config.times,
            max_steps=config.quoridor.max_steps,
        )

    def _run(self, model: LatestModel):
        t_name = f"time-{self.prefix}tournament"
        Timer.start(t_name)
        agent = create_alphazero(self.config, self.job_config.alphazero, overrides={"model_filename": model.filename})
        (
            _,
            relative_elo,
            win_perc,
            p1_stats,
            p2_stats,
            absolute_elo,
        ) = self.metrics.tournament(agent)
        elapsed = Timer.finish(t_name)

        metrics = {
            f"{self.prefix}relative_elo": relative_elo,
            f"{self.prefix}win_perc": win_perc,
            f"{self.prefix}absolute_elo": absolute_elo,
            t_name: elapsed,
            "Model version": model.version,
        }

        metrics.update(self.metrics.metrics_from_stats(self.prefix, p1_stats, p2_stats))

        self.wandb_run.log(metrics)
        self.upload_model_if_needed(metrics, model, agent.model_id())

        del agent


class DumbScoreBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: DumbScoreBenchmarkConfig, wandb_run):
        super().__init__(config, wandb_run, job_config.prefix)
        self.job_config = job_config
        self.metrics = Metrics(config.quoridor.board_size, config.quoridor.max_walls)

    def _run(self, latest: LatestModel):
        # We create the agent with temperature 0 for Dumb Score, since we want to see it in its best behavior.
        # In real playing, the temperature should be 0 or have dropped to 0 before getting to a terminal situation
        # like the ones in dumb score.
        agent = create_alphazero(
            self.config, self.job_config.alphazero, overrides={"model_filename": latest.filename, "temperature": 0}
        )

        t_name = f"time-{self.prefix}dumb_score"
        Timer.start(t_name)
        score = self.metrics.dumb_score(agent, verbose=False)
        elapsed = Timer.finish(t_name)

        metrics = {
            f"{self.prefix}dumb_score": score,
            t_name: elapsed,
            "Model version": latest.version,
        }
        self.wandb_run.log(metrics)
        self.upload_model_if_needed(metrics, latest, agent.model_id())

        del agent


class AgentEvolutionBenchmarkJob(BenchmarkJob):
    def __init__(self, config: Config, job_config: AgentEvolutionBenchmarkConfig, wandb_run):
        super().__init__(config, wandb_run, job_config.prefix)
        self.job_config = job_config
        params = AgentEvolutionTournamentParams(top_n=job_config.top_n, t=job_config.times)
        self.agent_evolution = AgentEvolutionTournament(
            config.quoridor.board_size,
            config.quoridor.max_walls,
            config.quoridor.max_steps,
            num_workers=0,
            params=params,
            verbose=False,
        )

    def _run(self, latest: LatestModel):
        overrides = {"model_filename": latest.filename, "nick": f"alphazero_{latest.version}"}

        agent_encoded_name = alphazero_encoded_name_from_config(self.config, self.job_config.alphazero, overrides)

        t_name = f"time-{self.prefix}agent_evolution"
        Timer.start(t_name)
        elos = self.agent_evolution.add_agent_and_compute(agent_encoded_name)
        elapsed = Timer.finish(t_name)

        metrics = {t_name: elapsed}
        elos_by_agent_episode = {}

        for nick, elo in elos.items():
            metrics[f"{self.prefix}agent_evolution_{nick}"] = int(elo)
            episode = int(nick.split("_")[-1])
            elos_by_agent_episode[episode] = int(elo)

        sorted_elos = sorted(elos_by_agent_episode.items(), key=lambda x: x[1], reverse=True)
        for i, (ep, _) in enumerate(sorted_elos):
            metrics[f"{self.prefix}agent_evolution_place_{i + 1}"] = ep
        self.wandb_run.log(metrics)

        # We don't call upload_model_if_needed here like the other benchmarks because it wouldn't work
        # since the name of the metrics change based on the model number, but even if we can find a workaround,
        # I don't think it would be useful.


def run_benchmark(config: Config, benchmark: BenchmarkScheduleConfig):
    LatestModel.wait_for_creation(config)
    freq = JobTrigger.from_string(config, benchmark.every)
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

    while not ShutdownSignal.is_set(config):
        # We get the model filename here so that all the jobs run with the same model
        latest = LatestModel.load(config)

        for job in jobs:
            job.run(latest)

        wandb_run.log({"Model version": latest.version}, commit=True)
        freq.wait(lambda: ShutdownSignal.is_set(config))


def create_benchmark_processes(config: Config) -> list[mp.Process]:
    ps = []
    for benchmark in config.benchmarks:
        p = mp.Process(target=run_benchmark, args=[config, benchmark])
        ps.append(p)

    return ps
