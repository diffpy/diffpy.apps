import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from diffpy.apps.app_agentify import agentify


@pytest.mark.parametrize(
    "args, expected_scope, expected_skill_dir",
    [
        # C1: diffpy.apps agentify
        #   Deploys workspace claude skill.
        #   Expect skill folder is created in the current working directory.
        (
            SimpleNamespace(
                agent="claude",
                system=False,
                update=False,
            ),
            "cwd",
            ".claude/skills/cmi-skill",
        ),
        # C2: diffpy.apps agentify --system
        #   Deploys system claude skill.
        #   Expect skill folder is created in the user's home directory.
        (
            SimpleNamespace(
                agent="claude",
                system=True,
                update=False,
            ),
            "home",
            ".claude/skills/cmi-skill",
        ),
        # C3: diffpy.apps agentify --agent codex
        #   Deploys workspace codex skill.
        #   Expect skill folder is created in the current working directory.
        (
            SimpleNamespace(
                agent="codex",
                system=False,
                update=False,
            ),
            "cwd",
            ".codex/skills/cmi-skill",
        ),
        # C4: diffpy.apps agentify --agent codex --system
        #   Deploys system codex skill.
        #   Expect skill folder is created in the user's home directory.
        (
            SimpleNamespace(
                agent="codex",
                system=True,
                update=False,
            ),
            "home",
            ".codex/skills/cmi-skill",
        ),
    ],
)
def test_agentify(args, expected_scope, expected_skill_dir):
    with tempfile.TemporaryDirectory() as tmp:
        with (
            mock.patch.object(Path, "home", return_value=Path(tmp) / "home"),
            mock.patch.object(Path, "cwd", return_value=Path(tmp) / "cwd"),
        ):
            agentify(args)
            expected_path = Path(tmp) / expected_scope / expected_skill_dir
            assert expected_path.exists()


def test_agentify_update():
    with tempfile.TemporaryDirectory() as tmp:
        with (
            mock.patch.object(Path, "home", return_value=Path(tmp) / "home"),
            mock.patch.object(Path, "cwd", return_value=Path(tmp) / "cwd"),
        ):
            # C1: Deploy again without --update flag when skill already exists.
            #  Expect FileExistsError to be raised, and the error message
            #  matches.
            args = SimpleNamespace(
                agent="claude",
                system=False,
                update=False,
            )
            agentify(args)
            skill_path = Path(tmp) / "cwd" / ".claude" / "skills" / "cmi-skill"
            assert skill_path.exists()
            args.update = True
            agentify(args)
            pytest.raises(
                FileExistsError,
                match=re.escape(
                    f"Agentic skill cmi-skill already exists at {skill_path}. "
                    "To overwrite, pass '--update' flag to update the skill"
                ),
            )
            # C2: Deploy again with --update flag when skill already exists
            #  with a dummy file in the skill directory.
            #  Expect no error to be raised, and the skill is updated.
            dummy_file = skill_path / "dummy.txt"
            dummy_file.touch()
            assert dummy_file.exists()
            args.update = True
            agentify(args)
            assert not dummy_file.exists()
