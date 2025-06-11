"""
AlphaZero wandb metrics logger.

This module provides a threaded logger that reads AlphaZero's learner.jsonl file
and sends the metrics to wandb.
"""

import json
import os
import threading
import time
from typing import Any, Dict, List


class AlphaZeroWandbLogger:
    """Logger that reads AlphaZero's learner.jsonl and uploads metrics to wandb.

    This class is designed to be run in a separate thread to monitor the
    learner.jsonl file and upload metrics to wandb as they become available.
    """

    def __init__(
        self,
        experiment_dir: str,
        wandb_run: Any,
        watch_interval_seconds: int = 30,
    ):
        """
        Initialize the logger.

        Args:
            experiment_dir: Directory where learner.jsonl is located
            wandb_run: Direct wandb run object to log metrics to
            watch_interval_seconds: How often to check for updates when watching
        """
        self.experiment_dir = experiment_dir
        self.jsonl_path = os.path.join(experiment_dir, "learner.jsonl")
        self.wandb_run = wandb_run
        self.watch_interval = watch_interval_seconds
        self.last_position = 0
        # wandb_run is already set in __init__
        self.stop_event = threading.Event()
        self.thread = None

    def _read_config(self) -> Dict[str, Any]:
        """Read the config.json file to get experiment configuration."""
        config_path = os.path.join(self.experiment_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: No config.json found at {config_path}")
            return {}

        try:
            with open(config_path, "r") as f:
                content = f.read().strip()
                if not content:  # Empty file
                    print(f"Warning: config.json at {config_path} is empty")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse config.json: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Error reading config.json: {e}")
            return {}

    def _parse_new_lines(self) -> List[Dict[str, Any]]:
        """Read new lines from the jsonl file."""
        if not os.path.exists(self.jsonl_path):
            return []

        entries = []
        try:
            with open(self.jsonl_path, "r") as f:
                f.seek(self.last_position)
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse line: {line}")
                self.last_position = f.tell()
        except FileNotFoundError:
            # File might not exist yet
            pass
        except Exception as e:
            print(f"Error reading from {self.jsonl_path}: {e}")
        return entries

    def _log_entries(self, entries: List[Dict[str, Any]]):
        """Log entries to wandb."""
        if not entries:
            return

        for entry in entries:
            # Extract step number
            step = entry.get("step", 0)

            # Convert all scalar values to a flattened dict
            metrics = {}
            for k, v in entry.items():
                if k != "step" and isinstance(v, (int, float)):
                    metrics[k] = v

            # Log metrics to wandb
            if metrics:
                self.wandb_run.log(metrics, step=step)

    def _watch_loop(self):
        """Main watch loop that monitors the file and uploads metrics."""
        # Read config but don't need to use it directly
        self._read_config()

        # Process existing data
        entries = self._parse_new_lines()
        if entries:
            self._log_entries(entries)

        # Watch for updates until stopped
        while not self.stop_event.is_set():
            time.sleep(self.watch_interval)

            # Check if the file exists now (might have been created after init)
            entries = self._parse_new_lines()
            if entries:
                self._log_entries(entries)

    def start(self):
        """Start the logger in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Logger is already running")
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._watch_loop, name="AlphaZeroWandbLogger", daemon=True)
        self.thread.start()
        print(f"Started wandb metrics logger for {self.jsonl_path}")

    def stop(self):
        """Stop the logger thread."""
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                print("Warning: Logger thread did not stop cleanly")
            else:
                print("Stopped wandb metrics logger")

        self.thread = None
