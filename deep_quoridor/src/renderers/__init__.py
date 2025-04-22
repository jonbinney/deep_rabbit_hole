from arena_utils import ArenaPlugin


class Renderer(ArenaPlugin):
    """
    Base class for all renderers, which take a game and render it in some way.
    """

    renderers = {}

    def __init_subclass__(cls, **kwargs):
        friendly_name = cls.__name__.replace("Renderer", "").lower()
        Renderer.renderers[friendly_name] = cls

    @staticmethod
    def create(friendly_name: str):
        return Renderer.renderers[friendly_name]()

    @staticmethod
    def names():
        return list(Renderer.renderers.keys())


__all__ = [
    "ArenaResults2Renderer",
    "ArenaResultsRenderer",
    "CursesBoardRenderer",
    "EloResultsRenderer",
    "MatchResultsRenderer",
    "NoneRenderer",
    "ProgressBarRenderer",
    "PygameRenderer",
    "Renderer",
    "TextBoardRenderer",
    "TrainingStatusRenderer",
]


from renderers.arena_results import ArenaResultsRenderer  # noqa: E402, F401
from renderers.arena_results2 import ArenaResults2Renderer  # noqa: E402, F401
from renderers.curses_board import CursesBoardRenderer  # noqa: E402, F401
from renderers.elo_results import EloResultsRenderer  # noqa: E402, F401
from renderers.match_results import MatchResultsRenderer  # noqa: E402, F401
from renderers.none import NoneRenderer  # noqa: E402, F401
from renderers.progress_bar import ProgressBarRenderer  # noqa: E402, F401
from renderers.pygame import PygameRenderer  # noqa: E402, F401
from renderers.text_board import TextBoardRenderer  # noqa: E402, F401
from renderers.training_status import TrainingStatusRenderer  # noqa: E402, F401
