import os
from pathlib import Path

from .utils.logging_utils import logger

# ---------------------------------------------------------------------------- #
# Setup paths
# ---------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"
# Non-package data
if os.getenv("MS2_REAL2SIM_ASSET_DIR") is not None:
    ASSET_DIR = Path(os.getenv("MS2_REAL2SIM_ASSET_DIR"))
elif os.path.exists(PACKAGE_DIR.parent.resolve() / "data"):
    # ManiSkill2_real2sim comes with a data directory, will try to find assets there
    ASSET_DIR = PACKAGE_DIR.parent.resolve() / "data"
elif os.getenv("MS2_ASSET_DIR") is not None:
    # if the original ManiSkill2 is co-installed, find the assets there
    ASSET_DIR = Path(os.getenv("MS2_ASSET_DIR"))
else:
    ASSET_DIR = Path("data")


def format_path(p: str):
    return p.format(
        PACKAGE_DIR=PACKAGE_DIR,
        PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR,
        ASSET_DIR=ASSET_DIR,
    )


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def get_commit_info(show_modified_files=False, show_untracked_files=False):
    """Get git commit information."""
    # isort: off
    import git

    try:
        repo = git.Repo(PACKAGE_DIR.parent)
    except git.InvalidGitRepositoryError as err:
        logger.warning("mani_skill2_real2sim is not installed with git.")
        return None
    else:
        commit_info = {}
        commit_info["commit_id"] = str(repo.head.commit)
        commit_info["branch"] = (
            None if repo.head.is_detached else repo.active_branch.name
        )

        if show_modified_files:
            # https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython
            modified_files = [item.a_path for item in repo.index.diff(None)]
            commit_info["modified"] = modified_files

        if show_untracked_files:
            untracked_files = repo.untracked_files
            commit_info["untracked"] = modified_files

        # https://github.com/gitpython-developers/GitPython/issues/718#issuecomment-360267779
        repo.__del__()
        return commit_info
