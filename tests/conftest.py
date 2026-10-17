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
    from diffpy.apps.parametric_refinement.parametric_model import (
        ParametricModelEquation,
    )

    submodel = ParametricModelEquation("sub", "a*x")
    model = ParametricModelEquation("main", "A*sin(u)")
    model.register_submodel(symbol="u", submodel=submodel)
    return model, submodel


@pytest.fixture
def sine_profile():
    xobs = numpy.linspace(-numpy.pi, numpy.pi, 100)
    yobs = numpy.sin(xobs) + 1e-3 * numpy.random.normal(size=xobs.shape)
    sine_profile = Profile()
    sine_profile.set_observed_profile(xobs, yobs)
    return sine_profile


@pytest.fixture
def mock_clone_skills(user_filesystem, mocker):
    def fake_clone(cmd, **kwargs):
        # the last argument is the temporary directory to clone into
        target_dir = Path(cmd[-1])
        (target_dir / "skills.json").write_text(
            json.dumps(["diffpy-playbook"])
        )
        skill_dir = target_dir / "diffpy-playbook"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("skill")

    mocker.patch("diffpy.apps.app_agentify.ensure_setup", return_value=True)
    mocker.patch(
        "diffpy.apps.app_agentify.subprocess.run", side_effect=fake_clone
    )
    mocker.patch.object(
        Path, "home", return_value=user_filesystem / "home_dir"
    )
    mocker.patch.object(Path, "cwd", return_value=user_filesystem / "cwd_dir")
    return user_filesystem
