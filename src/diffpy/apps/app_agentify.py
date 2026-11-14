import json
import shutil
import socket
import subprocess
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

REPO_URL = "https://github.com/diffpy/cmi-agent-skills"


def ensure_setup():
    def internet_available():
        try:
            socket.create_connection(("github.com", 443), timeout=3)
            return True
        except OSError:
            return False

    git_available = shutil.which("git") is not None
    agentify_available = internet_available() and git_available
    return agentify_available


@contextmanager
def cloned_skills():
    """Yield the path of a fresh clone of the skills repo."""
    if not ensure_setup():
        raise ValueError(
            "Internet connection or git unavailable. "
            "Please ensure git is installed and you have an active internet "
            f"connection to {REPO_URL}."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/skills.json",
            ],
            check=True,
            capture_output=True,
        )
        yield tmp_path


def list_skills(args):
    console = Console(
        theme=Theme(
            {
                "skill_name": "bold cyan",
                "key": "green",
                "value": "grey70",
            }
        )
    )

    def print_skill_brief(name, skills_brief_dict):
        line = Text(f"{name}: ", style="skill_name")
        if "description" not in skills_brief_dict[name]:
            skills_brief_dict[name][
                "description"
            ] = "Description not available."
        description = skills_brief_dict[name]["description"]
        line.append(str(description), style="value")
        console.print(line)

    def print_skill_details(name, skills_brief_dict):
        console.print(Text(name, style="skill_name"))
        fields = skills_brief_dict[name]
        width = max((len(key) for key in fields), default=0)
        for key, value in fields.items():
            line = Text(f"  {key:<{width}}  ", style="key")
            line.append(str(value), style="value")
            console.print(line)

    with cloned_skills() as tmp_path:
        skills_brief_dict = json.loads((tmp_path / "skills.json").read_text())
        if args.requested_skill_names is None:
            # No skills specified, print up to 10 skills
            n_skills_to_show = 10
            for i, name in enumerate(skills_brief_dict):
                if i >= n_skills_to_show:
                    break
                print_skill_brief(name, skills_brief_dict)
        else:
            # skills specified by the user.
            for name in args.requested_skill_names:
                if name not in skills_brief_dict:
                    warnings.warn(
                        f"Skill '{name}' not found. "
                        f"Run 'agentify list' to see available skills."
                    )
                    continue
                print_skill_details(name, skills_brief_dict)


def agentify(args):
    agent = args.agent
    system_flag = args.system
    # determine destination directory based on the agent and scope
    if agent == "claude":
        skills_dir = ".claude/skills"
    elif agent == "codex":
        skills_dir = ".codex/skills"
    else:
        raise KeyError(
            f"{agent} could not be recognized. "
            "Please use either 'claude' or 'codex'."
        )
    if system_flag:
        dest_dir = Path().home() / skills_dir
    else:
        dest_dir = Path().cwd() / skills_dir
    # clone the skills repository and deploy requested skills
    with cloned_skills() as tmp_path:
        skills_brief_dict = json.loads((tmp_path / "skills.json").read_text())
        if args.requested_skill_names is not None:
            unknown_names = [
                name
                for name in args.requested_skill_names
                if name not in skills_brief_dict
            ]
            if unknown_names:
                raise ValueError(
                    f"Skill(s) {unknown_names} not found. "
                    "Run 'agentify list' to see available skills."
                )
        for skill_name in skills_brief_dict:
            if (
                args.requested_skill_names is not None
                and skill_name not in args.requested_skill_names
            ):
                # skip when skill is not requested
                continue
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    f"{skill_name}/",
                ],
                check=True,
                capture_output=True,
            )
            destination = dest_dir / skill_name
            if destination.exists():
                if args.update:
                    shutil.rmtree(destination)
                    shutil.copytree(
                        tmp_path / skill_name, destination, dirs_exist_ok=True
                    )
                    print(f"Updated existing skill at {destination}")
                else:
                    warnings.warn(
                        f"Skill at {destination} already exists. "
                        "Use '--update' flag to overwrite."
                    )
                    continue
            else:
                shutil.copytree(
                    tmp_path / skill_name, destination, dirs_exist_ok=True
                )
                print(f"Skill {skill_name} has been deployed to {destination}")
