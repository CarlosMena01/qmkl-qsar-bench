import copy

import pytest

from qmkl_bench.config import load_config


@pytest.fixture(scope="session")
def base_cfg():
    return load_config("configs/default.yaml")


@pytest.fixture()
def cfg(base_cfg, tmp_path):
    c = copy.deepcopy(base_cfg)
    c["data"]["name"] = "synthetic"
    c["data"]["dir"] = str(tmp_path / "data")
    c["cache"]["dir"] = str(tmp_path / "cache")
    return c
