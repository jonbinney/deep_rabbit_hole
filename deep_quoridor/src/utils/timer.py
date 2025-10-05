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

    starts = {}
    counts = {}
    totals = {}
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
        cls.totals[name] = cls.totals.get(name, 0.0) + elapsed
        if episode is not None:
            print(f"TIMER: [{episode}] {name} took {format_time(elapsed)}")
            if cls.wandb_run:
                cls.wandb_run.log({f"time-{name}": elapsed, "Episode": episode})

    @classmethod
    def log_totals(cls):
        if len(cls.starts) > 0:
            print(f"TIMER: WARNING - timers for {list(cls.starts.keys())} still running")

        print("===== Timer Stats ====")
        for name, total in cls.totals.items():
            print(f"- {name} took {format_time(total)} in {cls.counts[name]} calls")
        print("======================")

        if cls.wandb_run:
            totals = {f"time-total-{name}": total for name, total in cls.totals.items()}
            print(totals)
            cls.wandb_run.log(totals)


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
