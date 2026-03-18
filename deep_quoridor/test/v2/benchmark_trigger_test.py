import time

import pytest
from v2.common import JobTrigger, ModelJobTrigger, TimeJobTrigger
from v2.config import Config, UserConfig
from v2.yaml_models import LatestModel


def _minimal_user_config():
    return UserConfig(
        run_id="test-trigger",
        quoridor={"board_size": 5, "max_walls": 2, "max_steps": 50},
        alphazero={"network": {"type": "mlp"}, "mcts_n": 100, "mcts_c_puct": 1.2},
        self_play={"num_workers": 1, "parallel_games": 1},
        training={
            "games_per_training_step": 10,
            "learning_rate": 0.001,
            "batch_size": 64,
            "weight_decay": 0.0001,
            "replay_buffer_size": 1000,
        },
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Prevent real sleeps in wait_for_creation and wait() loops."""
    monkeypatch.setattr(time, "sleep", lambda _: None)


@pytest.fixture
def config(tmp_path):
    user = _minimal_user_config()
    return Config.from_user(user, str(tmp_path))


@pytest.fixture
def config_at_model_0(config):
    """Config with latest.yaml written at version 0."""
    LatestModel.write(config, "model_0.pt", 0)
    return config


# ---------------------------------------------------------------------------
# A. from_string() parsing
# ---------------------------------------------------------------------------


class TestFromStringModels:
    def test_from_string_models(self, config_at_model_0):
        trigger = JobTrigger.from_string(config_at_model_0, "5 models")
        assert isinstance(trigger, ModelJobTrigger)
        assert trigger.every_model == 5

    def test_from_string_model_singular(self, config_at_model_0):
        trigger = JobTrigger.from_string(config_at_model_0, "1 model")
        assert isinstance(trigger, ModelJobTrigger)
        assert trigger.every_model == 1

    def test_from_string_whitespace(self, config_at_model_0):
        trigger = JobTrigger.from_string(config_at_model_0, "  5  models  ")
        assert isinstance(trigger, ModelJobTrigger)
        assert trigger.every_model == 5

    def test_from_string_case_insensitive(self, config_at_model_0):
        trigger = JobTrigger.from_string(config_at_model_0, "5 Models")
        assert isinstance(trigger, ModelJobTrigger)
        assert trigger.every_model == 5


class TestFromStringTime:
    def test_from_string_seconds(self, config):
        trigger = JobTrigger.from_string(config, "30 seconds")
        assert isinstance(trigger, TimeJobTrigger)
        assert trigger.every_s == 30

    def test_from_string_second_singular(self, config):
        trigger = JobTrigger.from_string(config, "1 second")
        assert isinstance(trigger, TimeJobTrigger)
        assert trigger.every_s == 1

    def test_from_string_minutes(self, config):
        trigger = JobTrigger.from_string(config, "5 minutes")
        assert isinstance(trigger, TimeJobTrigger)
        assert trigger.every_s == 300

    def test_from_string_hours(self, config):
        trigger = JobTrigger.from_string(config, "2 hours")
        assert isinstance(trigger, TimeJobTrigger)
        assert trigger.every_s == 7200


class TestFromStringErrors:
    def test_from_string_invalid_unit(self, config):
        with pytest.raises(ValueError, match="Invalid frequency string"):
            JobTrigger.from_string(config, "5 bananas")

    def test_from_string_zero(self, config_at_model_0):
        with pytest.raises(ValueError, match="positive integer"):
            JobTrigger.from_string(config_at_model_0, "0 models")

    def test_from_string_negative(self, config):
        with pytest.raises(ValueError, match="Invalid frequency string"):
            JobTrigger.from_string(config, "-1 models")

    def test_from_string_empty(self, config):
        with pytest.raises(ValueError, match="Invalid frequency string"):
            JobTrigger.from_string(config, "")


# ---------------------------------------------------------------------------
# B. ModelJobTrigger behavior
# ---------------------------------------------------------------------------


class TestModelJobTrigger:
    def test_init_sets_next_model(self, config_at_model_0):
        trigger = ModelJobTrigger(config_at_model_0, every_model=5)
        assert trigger.next_model == 5

    def test_is_ready_false_before_target(self, config_at_model_0):
        trigger = ModelJobTrigger(config_at_model_0, every_model=5)
        # Still at model 0, next_model is 5 → not ready
        assert trigger.is_ready() is False

    def test_is_ready_true_at_target(self, config_at_model_0):
        trigger = ModelJobTrigger(config_at_model_0, every_model=5)
        LatestModel.write(config_at_model_0, "model_5.pt", 5)
        assert trigger.is_ready() is True

    def test_is_ready_true_past_target(self, config_at_model_0):
        trigger = ModelJobTrigger(config_at_model_0, every_model=5)
        LatestModel.write(config_at_model_0, "model_7.pt", 7)
        assert trigger.is_ready() is True

    def test_wait_advances_from_previous_target(self, config_at_model_0):
        """
        THE BUG: when models are produced faster than benchmarks run,
        wait() should advance next_model relative to the previous target,
        not relative to the current model version.

        Scenario: every=5, init at model 0 → next_model=5.
        Models advance to 9 (simulating slow benchmarks).
        After wait(), next_model should be 10 (5+5), NOT 14 (9+5).
        """
        trigger = ModelJobTrigger(config_at_model_0, every_model=5)
        assert trigger.next_model == 5

        # Simulate trainer producing models while benchmark was running
        LatestModel.write(config_at_model_0, "model_9.pt", 9)

        # wait() should return immediately since 9 >= 5
        result = trigger.wait()
        assert result is True

        # next_model should be 10 (previous target 5 + every 5), NOT 14 (current 9 + 5)
        assert trigger.next_model == 10


# ---------------------------------------------------------------------------
# C. TimeJobTrigger behavior
# ---------------------------------------------------------------------------


class TestTimeJobTrigger:
    def test_is_ready_false_when_just_created(self):
        trigger = TimeJobTrigger(every_s=3600)
        assert trigger.is_ready() is False

    def test_is_ready_true_when_past(self):
        trigger = TimeJobTrigger(every_s=3600)
        trigger.next_time = time.time() - 1
        assert trigger.is_ready() is True

    def test_wait_returns_true(self):
        trigger = TimeJobTrigger(every_s=0)
        trigger.next_time = time.time() - 1
        result = trigger.wait()
        assert result is True

    def test_wait_advances_next_time(self):
        trigger = TimeJobTrigger(every_s=60)
        trigger.next_time = time.time() - 1
        before = time.time()
        trigger.wait()
        # next_time should be approximately now + 60
        assert trigger.next_time >= before + 60

    def test_wait_exits_on_should_exit(self):
        trigger = TimeJobTrigger(every_s=3600)
        result = trigger.wait(should_exit=lambda: True)
        assert result is False
