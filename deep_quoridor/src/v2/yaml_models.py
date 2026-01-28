import time

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from v2.config import Config


class LatestModel(BaseModel):
    filename: str
    version: int

    @classmethod
    def load(cls, config: Config):
        return parse_yaml_file_as(cls, config.paths.latest_model_yaml)

    @classmethod
    def write(cls, config: Config, filename: str, version: int):
        latest = LatestModel(filename=filename, version=version)
        to_yaml_file(config.paths.latest_model_yaml, latest)

    @classmethod
    def wait_for_creation(cls, config: Config, timeout: int = 60):
        start_time = time.time()
        while not config.paths.latest_model_yaml.exists():
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout: {config.paths.latest_model_yaml} not found after {timeout} seconds.")
        time.sleep(1)


class GameInfo(BaseModel):
    model_version: int
    game_length: int
    creator: str
