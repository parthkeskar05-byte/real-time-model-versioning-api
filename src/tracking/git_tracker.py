"""Git integration for tracking commit SHAs with MLflow runs."""

import git
from typing import Optional
import yaml


class GitTracker:
    """Track git commits for model reproducibility."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize git tracker.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.git_config = config['git']
        self.auto_tracking = self.git_config.get('auto_commit_tracking', True)

        try:
            self.repo = git.Repo(self.git_config.get('repo_path', '.'))
        except git.InvalidGitRepositoryError:
            self.repo = None

    def get_current_commit(self) -> Optional[str]:
        """Get the current git commit SHA.

        Returns:
            Commit SHA (short) or None if not in a git repo
        """
        if not self.repo:
            return None

        try:
            return self.repo.head.commit.hexsha[:7]
        except Exception:
            return None

    def get_commit_info(self, commit_sha: str) -> Optional[dict]:
        """Get detailed information about a commit.

        Args:
            commit_sha: Git commit SHA

        Returns:
            Dictionary with commit details or None
        """
        if not self.repo:
            return None

        try:
            commit = self.repo.commit(commit_sha)
            return {
                "sha": commit.hexsha[:7],
                "full_sha": commit.hexsha,
                "author": str(commit.author),
                "message": commit.message.strip(),
                "timestamp": commit.committed_datetime.isoformat()
            }
        except Exception:
            return None

    def is_repo_clean(self) -> bool:
        """Check if the repository has uncommitted changes.

        Returns:
            True if repo is clean, False otherwise
        """
        if not self.repo:
            return True

        return not self.repo.is_dirty()
