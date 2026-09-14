import json
from pathlib import Path

import numpy
import pytest

from diffpy.srfit.fitbase import (
    Profile,
)


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


@pytest.fixture
def nested_sine_model():
    from diffpy.apps.refinebase.parametric_model import ParametricModelEquation

    submodel = ParametricModelEquation("sub", "a*x")
    model = ParametricModelEquation("main", "A*sin(u)")
    model.register_submodel(symbol="u", submodel=submodel)
    return model, submodel


@pytest.fixture
def sine_profile():
    xobs = numpy.linspace(-numpy.pi, numpy.pi, 100)
    yobs = numpy.sin(xobs) + 1e-3 * numpy.random.normal(size=xobs.shape)
    sine_profile = Profile()
    sine_profile.setObservedProfile(xobs, yobs)
    return sine_profile
