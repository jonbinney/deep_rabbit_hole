"""Periodic AI-generated training reports.

When ``config.ai_report`` is set, ``train_v2`` spawns :func:`run_ai_reporter`
in a sibling process. This process:

1. On startup, generates a one-time *context document* (``reports/context.md``)
   by asking the AI to read the config and source code and explain the run's
   hyperparameters and project-specific metrics (e.g. ``dumb_score``).
   If ``reports/context.md`` already exists (resumed run), regeneration is
   skipped.

2. Loops on a :class:`~v2.common.JobTrigger` parsed from ``config.ai_report.every``.
   Each tick, it asks the AI to produce the next ``reports/report_NNN.md``,
   pointing the AI at the context doc, the previous report (for delta
   analysis), and a fresh wandb metrics snapshot. Report numbering continues
   from the highest existing ``report_*.md`` on disk, so resume works.

3. Each report is uploaded to wandb as an artifact of type ``ai_report``.

The AI is invoked as a subprocess (``claude -p <prompt>``) with the run
directory as the working directory, so it can read ``config.yaml``, the
metrics snapshot, and prior reports directly from the filesystem.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import wandb
from v2.common import JobTrigger, MockWandb, ShutdownSignal
from v2.config import Config
from v2.wandb_metrics import dump_group_metrics

SUPPORTED_AIS = ("claude",)

# How long to wait for the AI subprocess before killing it.
AI_SUBPROCESS_TIMEOUT_SECONDS = 20 * 60


# ---------------------------------------------------------------------------
# AI backend abstraction
# ---------------------------------------------------------------------------


class AIBackend(ABC):
    """Run an AI tool over a prompt."""

    name: str

    @abstractmethod
    def check_available(self) -> None:
        """Raise RuntimeError with a helpful message if the tool isn't installed."""

    @abstractmethod
    def generate(self, prompt: str, cwd: Path) -> None:
        """Invoke the AI with ``prompt``, running inside ``cwd``.

        The prompt must tell the AI where to write its output; this method does
        not return the AI's stdout.
        """

    @abstractmethod
    def generate_text(self, prompt: str, cwd: Path) -> str:
        """Invoke the AI with ``prompt`` inside ``cwd`` and return its stdout.

        Used for on-demand reports where the output goes back to the user
        instead of to disk.
        """


class ClaudeBackend(AIBackend):
    name = "claude"

    def __init__(self, model: Optional[str] = None):
        """``model``: optional Claude model identifier (e.g. 'sonnet', 'opus',
        'haiku', or a full model ID). ``None`` uses the CLI's default."""
        self.model = model

    def check_available(self) -> None:
        if shutil.which("claude") is None:
            raise RuntimeError(
                "AI reports are enabled (ai_report.ai='claude') but the 'claude' CLI "
                "was not found on PATH. Install Claude Code from "
                "https://claude.com/claude-code and make sure `claude` is runnable, "
                "or remove the 'ai_report' section from the config."
            )

    def _run(self, prompt: str, cwd: Path) -> subprocess.CompletedProcess:
        # --print (non-interactive) with --permission-mode acceptEdits so the AI
        # can read source files and (in file-writing mode) write reports without
        # interactive prompts.
        cmd = [
            "claude",
            "--print",
            "--permission-mode",
            "acceptEdits",
        ]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(prompt)
        model_label = self.model or "default"
        print(f"[ai_report] invoking claude (model={model_label}, cwd={cwd})")
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=AI_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def generate(self, prompt: str, cwd: Path) -> None:
        self._run(prompt, cwd)

    def generate_text(self, prompt: str, cwd: Path) -> str:
        return self._run(prompt, cwd).stdout


def backend_for(ai: str, model: Optional[str] = None) -> AIBackend:
    if ai == "claude":
        return ClaudeBackend(model=model)
    raise ValueError(f"Unknown ai_report.ai={ai!r}. Supported: {', '.join(SUPPORTED_AIS)}")


# Backward-compatible alias used inside this module.
_backend_for = backend_for


def check_ai_available(ai: str) -> None:
    """Raise RuntimeError with a helpful message if the requested AI isn't installed.

    Called from train_v2.py at startup, before any processes are spawned, so an
    incorrectly configured run aborts early instead of silently failing later.
    """
    backend_for(ai).check_available()


# ---------------------------------------------------------------------------
# Report numbering
# ---------------------------------------------------------------------------


REPORT_GLOB = "report_*.md"


def _report_path(reports_dir: Path, index: int) -> Path:
    return reports_dir / f"report_{index:03d}.md"


def _report_index(path: Path) -> Optional[int]:
    stem = path.stem  # "report_007"
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _next_report_index(reports_dir: Path) -> int:
    """1-indexed next report number.

    Uses ``max(existing index) + 1`` so gaps in the sequence never cause a
    report to be overwritten.
    """
    indices = [i for i in (_report_index(p) for p in reports_dir.glob(REPORT_GLOB)) if i is not None]
    return max(indices) + 1 if indices else 1


def _latest_report(reports_dir: Path) -> Optional[Path]:
    candidates = [(i, p) for p in reports_dir.glob(REPORT_GLOB) for i in [_report_index(p)] if i is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _source_paths_for_context(src_root: Path) -> list[Path]:
    """Source files the AI should read to understand the training pipeline."""
    return [
        src_root / "v2" / "config.py",
        src_root / "v2" / "trainer.py",
        src_root / "v2" / "benchmarks.py",
        src_root / "v2" / "self_play.py",
        src_root / "metrics.py",
        src_root / "agents" / "alphazero" / "mcts.py",
        src_root / "agents" / "alphazero" / "nn_evaluator.py",
        src_root / "agents" / "alphazero" / "resnet_network.py",
        src_root / "agents" / "alphazero" / "mlp_network.py",
    ]


def _build_context_prompt(config: Config, src_root: Path, context_md: Path) -> str:
    source_list = "\n".join(f"- {p}" for p in _source_paths_for_context(src_root))
    return f"""You are helping document a reinforcement-learning training run so that
future AI reports (generated during training) have the context they need to
interpret metrics correctly.

Please produce a single markdown file at:

    {context_md}

It must be self-contained: later reports will read only this file plus the
previous report, not the original source.

Read the run's config at:

    {config.paths.config_file}

And read these source files to understand what the training does and what the
metrics mean (especially the project-specific ones like `dumb_score`):

{source_list}

The markdown file should cover:

1. A short summary of this run's configuration — board size, walls, network
   type, MCTS settings, self-play and training hyperparameters, benchmark
   schedules. Call out any values that differ from what looks like the
   project's common defaults.
2. What each logged wandb metric means. In particular, explain `dumb_score`
   (see metrics.py) — what it measures, how it's computed, whether lower or
   higher is better. Also explain the tournament metrics (win_perc, relative_elo,
   absolute_elo, etc.), policy_loss / value_loss / total_loss, and model_lag.
3. How the wandb group is structured: a logical run_id '{config.run_id}' maps
   to a wandb group under project '{config.wandb.project if config.wandb else "(no wandb)"}'
   containing `<run_id>-training`, `<run_id>-benchmark-<idx>`, and
   `<run_id>-ai-report`. Explain the 'Model version' x-axis convention.
4. Any non-obvious project conventions you notice (e.g. the `every: N models`
   scheduling idiom, `raw_` prefix meaning, `dumb_score` scale 0-100 where
   lower is better).
5. **Tweakable config fields.** Enumerate every hyperparameter that future
   reports are allowed to recommend adjusting. Source of truth: the pydantic
   models in config.py (QuoridorConfig, AlphaZeroBaseConfig, MLPConfig /
   ResnetConfig, SelfPlayConfig, AlphaZeroSelfPlayConfig, TrainingConfig, and
   the benchmark configs). For each field give: the dotted path
   (e.g. `training.learning_rate`), the type / valid range, and a one-line
   note on what it controls. Be exhaustive — this list is the authoritative
   reference future reports use when recommending hyperparameter changes, and
   they are explicitly forbidden from suggesting anything outside it.

Keep it tight — aim for something a future report can skim in under a minute
(except section 5, which can be a longer reference table).
Do not write anything outside the markdown file. When you're done, just confirm
with the path you wrote."""


def _build_report_prompt(
    config: Config,
    context_md: Path,
    prev_report: Optional[Path],
    report_path: Path,
    metrics_snapshot: Path,
) -> str:
    prev_line = (
        f"Previous report (for comparing progress):\n    {prev_report}\n"
        if prev_report is not None
        else "This is the first report for this run — there is no previous report.\n"
    )
    return f"""You are generating a periodic AI report on an in-progress training run.

Read first:

    {context_md}

{prev_line}
A fresh wandb metrics snapshot for this run's group has been written to:

    {metrics_snapshot}

It is a JSON dump with the shape:
    {{ "project", "group", "runs": {{ <wandb run name>: {{ "summary", "history", ... }} }} }}
Use it as your primary data source. You may also read the run's config at
{config.paths.config_file} and any files under {config.paths.run_dir} if useful.

Write the report to:

    {report_path}

The report MUST contain these sections, in this order:

1. **General observations about training progress.** How are losses evolving?
   How are benchmark metrics (win_perc, relative_elo, dumb_score) trending?
   How many models trained so far? Any anomalies?
2. **Progress since the last report.** If there is no previous report, say so.
   Otherwise, concretely compare: what changed in the metrics, did progress
   continue / stall / regress? Reference specific numbers.
3. **Likelihood this run produces a strong model.** Either state
   "too early to determine" (with a sentence on what you'd need to see to
   commit to a number) or give a score 0-10 with a one-paragraph justification.
4. **Is it worth continuing this run?** Especially if the likelihood is low,
   argue whether continuing will at least yield useful signal for tuning
   hyperparameters for the next run, or whether it's better to stop now.
5. **Recommended hyperparameter adjustments for the next run.**

   CRITICAL CONSTRAINT: every recommendation in this section MUST correspond to
   a field that already exists in the project's config schema. The context doc
   you read above enumerates the authoritative list. You can also cross-check
   against the pydantic models at {_src_root() / "v2" / "config.py"}, or the
   run's `config.yaml` at {config.paths.config_file}. Reference each suggestion
   by its dotted config path (e.g. `training.learning_rate`,
   `alphazero.mcts_n`, `alphazero.network.num_channels`,
   `self_play.alphazero.mcts_noise_epsilon`, `training.batch_size`).
   If a value must be an existing enum / literal (e.g. `alphazero.network.type`
   is `mlp` or `resnet`), respect that.
   DO NOT suggest anything that requires code changes here (no new schedulers,
   no new loss functions, no new network types — those go in section 6).
   Each suggestion gets one line of rationale tied to a metric you observed.

6. **(Optional) Suggested code changes / new features.** ONLY include this
   section if there's something concrete worth implementing — e.g. "add cosine
   learning-rate decay", "support a different optimizer", "expose a new MCTS
   parameter". Each item: what to implement, why the metrics suggest it would
   help, and (if relevant) which file would change. If nothing concrete comes
   to mind, OMIT this section entirely — do not fill it with speculation.

Keep the report readable (headings, short paragraphs, bullets where helpful).
Be honest about uncertainty. Do not write anything outside {report_path.name}."""


def _build_on_demand_prompt(
    project: str,
    group: str,
    metrics_snapshot: Path,
    repo_root: Path,
    guidance: Optional[str] = None,
) -> str:
    """Prompt for the on-demand CLI path: no file outputs, just stdout text."""
    sources = [
        "deep_quoridor/src/v2/config.py",
        "deep_quoridor/src/v2/trainer.py",
        "deep_quoridor/src/v2/benchmarks.py",
        "deep_quoridor/src/metrics.py",
    ]
    source_list = "\n".join(f"- {repo_root / p}" for p in sources)

    guidance_block = ""
    if guidance:
        guidance_block = (
            "\nSPECIFIC GUIDANCE FROM THE USER — address this directly and prominently "
            "in your report, in addition to the standard sections below:\n\n"
            f"{guidance}\n"
        )

    return f"""You are generating an ad-hoc AI report on a training run for the
deep_quoridor Quoridor RL project.

A wandb metrics snapshot for the group '{group}' in project '{project}' has been
written to:

    {metrics_snapshot}

It is a JSON dump with shape:
    {{ "project", "group", "runs": {{ <wandb run name>: {{ "config", "summary", "history", ... }} }} }}

Each run's "config" field holds wandb's native config dict (equivalent to the
training config.yaml). Use the snapshot as your primary data source — it is
self-contained, even for old runs with no local files.

To understand what the metrics mean and the project's conventions, read these
source files (absolute paths):
{source_list}

Key conventions to note:
- `dumb_score` is a 0-100 metric (lower = better) — see metrics.py.
- Tournament metrics use a `{{prefix}}` prefix (`raw_`, '', etc.); common keys
  are `{{prefix}}win_perc`, `{{prefix}}relative_elo`, `{{prefix}}absolute_elo`.
- The "Model version" axis increments once per trained model.
- A logical run_id maps to a wandb group with `<run_id>-training`,
  `<run_id>-benchmark-<idx>`, and optionally `<run_id>-ai-report`.
{guidance_block}
IMPORTANT: respond with the report as your final text answer. Do NOT write any
files — the user is capturing your stdout and printing it.

Produce a markdown report with these sections (adapted for ad-hoc use):

1. **Run summary.** One paragraph: what this run was configured to do
   (board size, walls, network, MCTS, self-play, key hyperparameters),
   and how far it got.
2. **General observations about training progress.** Loss trajectories,
   benchmark trends, dumb_score evolution, anomalies. Cite specific numbers.
3. **Likelihood this run produced a strong model.** Either "too early to
   determine" (with what you'd want to see), or a 0-10 score with a
   one-paragraph justification.
4. **Was/is it worth continuing this run?** Even if the answer is no, call out
   what signal (if any) it yields for tuning the next run.
5. **Recommended hyperparameter adjustments for the next run.**

   CRITICAL CONSTRAINT: every recommendation in this section MUST correspond to
   a field that already exists in the project's config schema — read
   {repo_root / "deep_quoridor/src/v2/config.py"} for the authoritative list of
   tweakable fields. Reference each suggestion by its dotted config path
   (e.g. `training.learning_rate`, `alphazero.mcts_n`,
   `alphazero.network.num_channels`, `self_play.alphazero.mcts_noise_epsilon`,
   `training.batch_size`). If a value must be an existing enum / literal
   (e.g. `alphazero.network.type` is `mlp` or `resnet`), respect that.
   DO NOT suggest anything that requires code changes here (no new schedulers,
   no new loss functions — those go in section 6).
   Each suggestion: config path, proposed value (or direction), one-line
   rationale tied to a metric you observed.

6. **(Optional) Suggested code changes / new features.** ONLY include this
   section if there's something concrete worth implementing — e.g. "add cosine
   learning-rate decay", "support a different optimizer", "expose a new MCTS
   parameter". Each item: what to implement, why the metrics suggest it would
   help, and (if relevant) which file would change. If nothing concrete comes
   to mind, OMIT this section entirely — do not fill it with speculation.

Be honest about uncertainty. Keep it readable."""


def generate_on_demand_report(
    project: str,
    group: str,
    ai: str = "claude",
    entity: Optional[str] = None,
    guidance: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate an AI training report on demand and return it as text.

    Does not write any files to disk and does not upload anything. Fetches
    wandb metrics for the group, invokes the AI, and returns its response.

    Args:
        project: wandb project name.
        group: wandb group name (equivalent to the logical run_id).
        ai: AI backend to use. Currently only "claude".
        entity: wandb entity (user/team), optional.
        guidance: Optional extra text appended to the prompt. Useful for asking
            the AI specific questions or steering the analysis.
        model: Optional AI model identifier (e.g. 'sonnet', 'opus'). None uses
            the backend's default.

    Returns:
        The AI's response text — a markdown report.
    """
    backend = backend_for(ai, model=model)
    backend.check_available()

    # ai_report.py lives at <repo>/deep_quoridor/src/v2/ai_report.py
    # -> parents: v2 -> src -> deep_quoridor -> <repo root>
    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    with tempfile.TemporaryDirectory(prefix="ai_report_on_demand_") as tmp:
        metrics_snapshot = Path(tmp) / "metrics.json"
        dump_group_metrics(
            project=project,
            group=group,
            out_path=metrics_snapshot,
            entity=entity,
        )
        prompt = _build_on_demand_prompt(
            project=project,
            group=group,
            metrics_snapshot=metrics_snapshot,
            repo_root=repo_root,
            guidance=guidance,
        )
        # cwd = repo root so the AI can freely read source files; the metrics
        # snapshot is passed by absolute path.
        return backend.generate_text(prompt, cwd=repo_root)


# ---------------------------------------------------------------------------
# wandb plumbing
# ---------------------------------------------------------------------------


def _init_wandb_run(config: Config):
    """Initialize the ai_report wandb run (or a mock if wandb is disabled)."""
    if not config.wandb:
        return MockWandb()
    run_id = f"{config.run_id}-ai-report"
    wandb_run = wandb.init(
        project=config.wandb.project,
        job_type="ai_report",
        group=config.run_id,
        name=run_id,
        id=run_id,
        resume="allow",
    )
    wandb.define_metric("Model version", hidden=True)
    wandb.define_metric("*", "Model version")
    return wandb_run


def _upload_report(wandb_run, config: Config, report_path: Path, index: int) -> None:
    """Upload a report as a wandb artifact. Silent no-op if wandb is mocked."""
    if isinstance(wandb_run, MockWandb):
        return
    try:
        # Prefix the artifact name with the project so that two runs with the
        # same run_id in different projects don't land on the same artifact.
        artifact_name = f"{config.wandb.project}-{config.run_id}-ai-report"
        artifact = wandb.Artifact(
            artifact_name,
            type="ai_report",
            metadata={
                "report_index": index,
                "run_id": config.run_id,
                "project": config.wandb.project,
            },
        )
        artifact.add_file(local_path=str(report_path))
        wandb_run.log_artifact(artifact, aliases=[f"r{index:03d}", "latest"])
        print(f"[ai_report] uploaded {report_path.name} as artifact {artifact_name}:r{index:03d}")
    except Exception as e:
        print(f"[ai_report] !!! failed to upload report artifact: {e}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _src_root() -> Path:
    # This file lives at deep_quoridor/src/v2/ai_report.py
    return Path(__file__).resolve().parent.parent


def _ensure_context_doc(config: Config, backend: AIBackend) -> Path:
    context_md = config.paths.reports / "context.md"
    if context_md.exists():
        print(f"[ai_report] reusing existing context doc at {context_md}")
        return context_md

    print(f"[ai_report] generating context doc at {context_md}")
    prompt = _build_context_prompt(config, _src_root(), context_md)
    backend.generate(prompt, cwd=config.paths.run_dir)
    if not context_md.exists():
        raise RuntimeError(f"AI did not produce {context_md} as instructed. Aborting periodic reports.")
    return context_md


def _generate_one_report(
    config: Config,
    backend: AIBackend,
    context_md: Path,
    wandb_run,
) -> Optional[Path]:
    index = _next_report_index(config.paths.reports)
    report_path = _report_path(config.paths.reports, index)
    prev_report = _latest_report(config.paths.reports)

    metrics_snapshot = config.paths.reports / f".metrics_{index:03d}.json"
    if config.wandb:
        try:
            dump_group_metrics(
                project=config.wandb.project,
                group=config.run_id,
                out_path=metrics_snapshot,
            )
        except Exception as e:
            print(f"[ai_report] !!! failed to dump wandb metrics: {e}")
            # We still attempt the report; the AI can fall back to reading files.
    else:
        metrics_snapshot.write_text('{"note": "wandb disabled — no metrics snapshot"}')

    prompt = _build_report_prompt(
        config=config,
        context_md=context_md,
        prev_report=prev_report,
        report_path=report_path,
        metrics_snapshot=metrics_snapshot,
    )

    print(f"[ai_report] generating {report_path.name}")
    try:
        backend.generate(prompt, cwd=config.paths.run_dir)
    except subprocess.TimeoutExpired:
        print(f"[ai_report] !!! AI subprocess timed out for {report_path.name}")
        return None
    except Exception as e:
        print(f"[ai_report] !!! AI subprocess failed: {e}")
        return None

    if not report_path.exists():
        print(f"[ai_report] !!! AI did not produce {report_path} — skipping upload")
        return None

    _upload_report(wandb_run, config, report_path, index)
    return report_path


def run_ai_reporter(config: Config) -> None:
    """Sibling-process entry point. Runs until ShutdownSignal is set."""
    if config.ai_report is None:
        return

    try:
        backend = _backend_for(config.ai_report.ai, model=config.ai_report.model)
    except Exception as e:
        print(f"[ai_report] cannot start reporter: {e}")
        return

    # Don't start racing before the trainer has even written its first model.
    from v2.yaml_models import LatestModel

    LatestModel.wait_for_creation(config)

    wandb_run = _init_wandb_run(config)

    try:
        context_md = _ensure_context_doc(config, backend)
    except Exception as e:
        print(f"[ai_report] failed to prepare context doc: {e}")
        traceback.print_exc()
        return

    trigger = JobTrigger.from_string(config, config.ai_report.every)

    # Unlike benchmarks, we do NOT run a report at t=0: a baseline report would
    # have nothing to analyze yet. Wait for the first trigger interval to elapse
    # before generating the first report.
    while trigger.wait(lambda: ShutdownSignal.is_set(config)):
        try:
            _generate_one_report(config, backend, context_md, wandb_run)
        except Exception as e:
            # One bad report shouldn't kill the whole reporter.
            print(f"[ai_report] !!! unexpected error during report generation: {e}")
            traceback.print_exc()

    print("[ai_report] shutdown signal received, exiting")
