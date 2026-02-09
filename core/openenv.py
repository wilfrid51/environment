"""OpenEnv shared protocol models.

This module is intended to be usable in two contexts:
- Host-side Python (normal package).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class OpenEnvResponse:
    """OpenEnv unified response for reset/step/state.

    This is the standard return type for all OpenEnv methods.
    HTTP layer will automatically serialize to JSON.
    """
    observation: str
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    episode_id: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "info": self.info,
        }