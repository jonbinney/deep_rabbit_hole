from agents.nn.cnn3c_v1 import Cnn3cV1Network
from agents.nn.cnn_v1 import CnnV1Network
from agents.nn.cnn_v2 import CnnV2Network
from agents.nn.cnn_v3 import CnnV3Network
from agents.nn.cnn_v4 import CnnV4Network
from agents.nn.flat_1024 import Flat1024Network
from agents.nn.pyramid_512_dropout import P512DropoutNetwork
from agents.nn.pyramid_1024_dropout import P1024DropoutNetwork

# Simply importing the classes will trigger the registration due to the BaseNN.__init_subclass__ method

__all__ = [
    "CnnV1Network",
    "CnnV2Network",
    "CnnV3Network",
    "CnnV4Network",
    "Cnn3cV1Network",
    "Flat1024Network",
    "P512DropoutNetwork",
    "P1024DropoutNetwork",
]
