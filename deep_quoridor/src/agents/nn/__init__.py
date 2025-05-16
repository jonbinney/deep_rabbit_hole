from agents.nn.cnn_v1 import CnnV1Network
from agents.nn.cnn_v2 import CnnV2Network
from agents.nn.cnn_v3 import CnnV3Network
from agents.nn.flat_1024 import Flat1024Network
from agents.nn.pyramid_512_dropout import P512DropoutNetwork

# Simply importing the classes will trigger the registration due to the BaseNN.__init_subclass__ method

__all__ = [
    "CnnV1Network",
    "CnnV2Network",
    "CnnV3Network",
    "Flat1024Network",
    "P512DropoutNetwork",
]
