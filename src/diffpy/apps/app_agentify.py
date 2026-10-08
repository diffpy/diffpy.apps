import shutil
import socket
import subprocess
import tempfile
import warnings
from pathlib import Path

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


def agentify(args):
    if not ensure_setup():
        raise ValueError(
            "Internet connection or git unavailable. "
            "Please ensure git is installed and you have an active internet "
            f"connection to {REPO_URL}."
        )
    agent = args.agent
    system_flag = args.system
    if agent == "claude":
        skills_dir = ".claude/skills"
    elif agent == "codex":
        skills_dir = ".codex/skills"
    if system_flag:
        dest_dir = Path().home() / skills_dir
    else:
        dest_dir = Path().cwd() / skills_dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            ["git", "clone", REPO_URL, str(tmp_path)],
            check=True,
        )
        for item in tmp_path.iterdir():
            if item.is_dir():
                destination = dest_dir / item.name
                if destination.exists():
                    if args.update:
                        shutil.rmtree(destination)
                        shutil.copytree(item, destination, dirs_exist_ok=True)
                        print(f"Updated existing skill at {destination}")
                    else:
                        warnings.warn(
                            f"Skill at {destination} already exists. "
                            "Use '--update' flag to overwrite."
                        )
                        continue
                else:
                    shutil.copytree(item, destination, dirs_exist_ok=True)
                    print(
                        f"Skill {item.name} has been deployed to {destination}"
                    )
