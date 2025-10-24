import time
from typing import Optional

import wandb
import wandb.wandb_run

from utils.misc import format_time


class Timer:
    """
    Class to measure elapsed time and log it to the console and optionally wandb.
    It's a class just to create a container for the functions and variables, it can't be instantiated.
    TO DO: would be nice if it would easily support multi-process, so we can gather information coming
    from different processes and compile it.
    """

    # Used to keep track of named time intervals and their sums
    starts = {}
    counts = {}
    total_times = {}

    # Used to keep track of cumulative values, e.g. "total number of inputs evaluated so far"
    counters = {}

    wandb_run: Optional[wandb.wandb_run.Run] = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("Timer is a static class and cannot be instantiated.")

    @classmethod
    def set_wandb_run(cls, wandb_run: wandb.wandb_run.Run):
        cls.wandb_run = wandb_run

    @classmethod
    def start(cls, name: str):
        if name in cls.starts:
            print(f"TIMER: WARNING - timer for {name} was already started")

        cls.starts[name] = time.perf_counter()

    @classmethod
    def finish(cls, name: str, episode: Optional[int] = None):
        if name not in cls.starts:
            print(f"TIMER: WARNING - timer for {name} was not started but trying to finish")
            return

        elapsed = time.perf_counter() - cls.starts[name]
        del cls.starts[name]

        cls.counts[name] = cls.counts.get(name, 0) + 1
        cls.total_times[name] = cls.total_times.get(name, 0.0) + elapsed
        if episode is not None:
            print(f"TIMER: [{episode}] {name} took {format_time(elapsed)}")
            if cls.wandb_run:
                cls.wandb_run.log({f"time-{name}": elapsed, "Episode": episode})

    @classmethod
    def log_totals(cls, episode: Optional[int] = None):
        if len(cls.starts) > 0:
            print(f"TIMER: WARNING - timers for {list(cls.starts.keys())} still running")

        print("===== Timer Stats ====")
        for name, total in cls.total_times.items():
            print(f"- {name} took {format_time(total)} in {cls.counts[name]} calls")
        print("===== Counter Stats ====")
        for name, total in cls.counters.items():
            print(f"- {name} counter is {total}")
        print("======================")

        if cls.wandb_run:
            total_times = {f"time-total-{name}": total for name, total in cls.total_times.items()}
            counters = {f"counter-{name}": total for name, total in cls.counters.items()}
            if episode is not None:
                total_times["Episode"] = episode
                counters["Episode"] = episode
            cls.wandb_run.log(total_times)
            cls.wandb_run.log(counters)

    @classmethod
    def increment_counter(cls, name: str, increment: int = 1, episode: Optional[int] = None):
        if name in cls.counters:
            cls.counters[name] += increment
        else:
            cls.counters[name] = increment

        if episode is not None:
            if cls.wandb_run:
                cls.wandb_run.log({f"counter-{name}": cls.counters[name], "Episode": episode})

    @classmethod
    def set_counter(cls, name: str, value: int, episode: Optional[int] = None):
        cls.counters[name] = value

        if episode is not None:
            if cls.wandb_run:
                cls.wandb_run.log({f"counter-{name}": cls.counters[name], "Episode": episode})


def timer(name: str):
    """
    Decorator to time a function or method
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            Timer.start(name)
            result = func(*args, **kwargs)
            Timer.finish(name)
            return result

        return wrapper

    return decorator
