import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffpy.apps.app_agentify import agentify, list_skills


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
                requested_skill_names=None,
            ),
            "cwd",
            ".claude/skills/diffpy-playbook",
        ),
        # C2: diffpy.apps agentify --system
        #   Deploys system claude skill.
        #   Expect skill folder is created in the user's home directory.
        (
            SimpleNamespace(
                agent="claude",
                system=True,
                update=False,
                requested_skill_names=None,
            ),
            "home",
            ".claude/skills/diffpy-playbook",
        ),
        # C3: diffpy.apps agentify --agent codex
        #   Deploys workspace codex skill.
        #   Expect skill folder is created in the current working directory.
        (
            SimpleNamespace(
                agent="codex",
                system=False,
                update=False,
                requested_skill_names=None,
            ),
            "cwd",
            ".codex/skills/diffpy-playbook",
        ),
        # C4: diffpy.apps agentify --agent codex --system
        #   Deploys system codex skill.
        #   Expect skill folder is created in the user's home directory.
        (
            SimpleNamespace(
                agent="codex",
                system=True,
                update=False,
                requested_skill_names=None,
            ),
            "home",
            ".codex/skills/diffpy-playbook",
        ),
        # C5: diffpy.apps agentify --skill diffpy-playbook
        #   Deploys the requested workspace claude skill.
        #   Expect skill folder is created in the current working directory.
        (
            SimpleNamespace(
                agent="claude",
                system=False,
                update=False,
                requested_skill_names=["diffpy-playbook"],
            ),
            "cwd",
            ".claude/skills/diffpy-playbook",
        ),
    ],
)
def test_agentify(args, expected_scope, expected_skill_dir, mock_clone_skills):
    agentify(args)
    scope_dir = {"cwd": "cwd_dir", "home": "home_dir"}[expected_scope]
    expected_path = mock_clone_skills / scope_dir / expected_skill_dir
    assert expected_path.exists()


@pytest.mark.parametrize(
    "requested_skill_names, expected_output",
    [
        # C1: diffpy.apps agentify list
        #   No skill name provided.
        #   Expect the brief listing without warnings.
        (None, "diffpy-playbook: Playbook skill."),
        # C2: diffpy.apps agentify list --skill diffpy-playbook
        #   A valid skill name provided.
        #   Expect the detailed listing without warnings.
        (
            ["diffpy-playbook"],
            "diffpy-playbook\n  description  Playbook skill.",
        ),
    ],
)
def test_list_skills(
    requested_skill_names,
    expected_output,
    mock_clone_skills,
    capsys,
    recwarn,
):
    args = SimpleNamespace(requested_skill_names=requested_skill_names)
    list_skills(args)
    assert expected_output in capsys.readouterr().out
    assert len(recwarn) == 0


def test_list_skills_bad(mock_clone_skills):
    # C1: diffpy.apps agentify list --skill wrong-skill
    #   An unknown skill name provided.
    #   Expect a warning that points the user to 'agentify list'.
    args = SimpleNamespace(requested_skill_names=["wrong-skill"])
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Skill 'wrong-skill' not found. "
            "Run 'agentify list' to see available skills."
        ),
    ):
        list_skills(args)


def test_agentify_bad(mock_clone_skills):
    # C1: diffpy.apps agentify --skill wrong-skill
    #   An unknown skill name provided.
    #   Expect ValueError that points the user to 'agentify list',
    #   and no skill is deployed.
    args = SimpleNamespace(
        agent="claude",
        system=False,
        update=False,
        requested_skill_names=["wrong-skill"],
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Skill(s) ['wrong-skill'] not found. "
            "Run 'agentify list' to see available skills."
        ),
    ):
        agentify(args)
    assert not (mock_clone_skills / "cwd_dir" / ".claude").exists()


def test_agentify_update(mock_clone_skills):
    tmp = mock_clone_skills
    # C1: Deploy again without --update flag when skill already exists.
    #  Expect FileExistsError to be raised, and the error message
    #  matches.
    args = SimpleNamespace(
        agent="claude",
        system=False,
        update=False,
        requested_skill_names=None,
    )
    agentify(args)
    skill_path = (
        Path(tmp) / "cwd_dir" / ".claude" / "skills" / "diffpy-playbook"
    )
    assert skill_path.exists()
    args.update = True
    agentify(args)
    pytest.raises(
        FileExistsError,
        match=re.escape(
            "Agentic skill diffpy-playbook already exists at "
            f"{skill_path}. "
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
