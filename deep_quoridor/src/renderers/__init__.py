from arena import ArenaPlugin


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


from renderers.arena_results import ArenaResultsRenderer  # noqa: E402, F401
from renderers.curses_board import CursesBoardRenderer  # noqa: E402, F401
from renderers.match_results import MatchResultsRenderer  # noqa: E402, F401
from renderers.none import NoneRenderer  # noqa: E402, F401
from renderers.progress_bar import ProgressBarRenderer  # noqa: E402, F401
from renderers.text_board import TextBoardRenderer  # noqa: E402, F401
from renderers.pygame import PygameRenderer  # noqa: E402, F401
