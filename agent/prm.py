"""
Process Reward Model (PRM) Wrapper

Loads trained PRM model and provides inference for stopping decisions.
Primary interface: should_stop_from_dict() / get_reward_from_features_dict() with GraphFeatures (20-dim).
Legacy interface: should_stop() / get_reward() with StateFeatures — deprecated.
"""

import warnings

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from .state import StateFeatures


class RewardModel(nn.Module):
    """
    MLP-based Process Reward Model.

    Architecture matches Stage 4 training:
    - LayerNorm + ReLU + Dropout for each hidden layer
    - Single scalar output
    """

    def __init__(self, input_dim: int = 17, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer: single reward score
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Return reward score"""
        return self.network(x).squeeze(-1)


class PRM:
    """PRM decision maker for search stopping"""

    def __init__(
        self,
        model_path: str,
        min_steps: int = 3,
        decline_threshold: float = 0.3,
        convergence_threshold: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize PRM.

        The PRM model was trained with Bradley-Terry preference loss, so rewards
        represent relative quality (higher = better stopping point) rather than
        absolute probabilities. Stopping decisions are based on:
        1. Decline detection: reward dropping suggests we've passed optimal
        2. Convergence: small reward changes suggest saturation

        Args:
            model_path: Path to trained model checkpoint
            min_steps: Minimum steps before considering stopping
            decline_threshold: Stop if reward drops by this amount
            convergence_threshold: Stop if reward change is below this
            device: Device to run inference on
        """
        self.min_steps = min_steps
        self.decline_threshold = decline_threshold
        self.convergence_threshold = convergence_threshold
        self.device = device

        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model.eval()

        # Track reward history per search
        self.reward_history: List[float] = []
        self.max_reward_seen: float = float('-inf')

    def _load_model(self, model_path: str) -> Tuple[RewardModel, dict]:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract config
        if isinstance(checkpoint, dict):
            feature_dim = checkpoint.get("feature_dim", 17)
            hidden_dims = checkpoint.get("hidden_dims", [128, 64, 32])
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            config = {
                "feature_dim": feature_dim,
                "hidden_dims": hidden_dims,
                "feature_names": checkpoint.get("feature_names", []),
                "feature_norms": checkpoint.get("feature_norms", {}),
                "best_val_acc": checkpoint.get("best_val_acc", 0)
            }
        else:
            # Raw state dict
            feature_dim = 17
            hidden_dims = [128, 64, 32]
            state_dict = checkpoint
            config = {"feature_dim": feature_dim, "hidden_dims": hidden_dims}

        # Create model with correct architecture
        model = RewardModel(input_dim=feature_dim, hidden_dims=hidden_dims)
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model, config

    def reset(self):
        """Reset history for new search"""
        self.reward_history = []
        self.max_reward_seen = float('-inf')

    def get_reward(self, features: StateFeatures, normalize: bool = True) -> float:
        """[LEGACY] Compute reward from StateFeatures.

        Prefer get_reward_from_features_dict() with GraphFeatures.to_dict().
        Raises ValueError if checkpoint was trained on GraphFeatures.
        """
        ckpt_names = self.config.get("feature_names", [])
        if ckpt_names:
            state_keys = set(features.to_dict().keys())
            if not set(ckpt_names).issubset(state_keys):
                raise ValueError(
                    f"Checkpoint expects GraphFeatures ({len(ckpt_names)}-dim) but received StateFeatures. "
                    "Use get_reward_from_features_dict() / should_stop_from_dict() with GraphFeatures.to_dict()."
                )
            return self.get_reward_from_features_dict(features.to_dict(), normalize=normalize)

        vec = features.to_normalized_array()
        expected_dim = self.config.get("feature_dim", len(vec))
        if len(vec) != expected_dim:
            raise ValueError(
                f"StateFeatures dimension ({len(vec)}) != checkpoint feature_dim ({expected_dim})."
            )

        with torch.no_grad():
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw_reward = self.model(x).item()
            reward = 1 / (1 + np.exp(-raw_reward)) if normalize else raw_reward

        self.reward_history.append(reward)
        return reward

    def get_reward_from_dict(self, state_dict: dict) -> float:
        """
        Compute reward from state dictionary.

        Args:
            state_dict: Dictionary with state features

        Returns:
            Reward score
        """
        features = StateFeatures(**state_dict)
        return self.get_reward(features)

    def get_reward_from_features_dict(self, features_dict: dict, normalize: bool = True) -> float:
        """
        Compute reward from a raw feature dict (e.g. GraphFeatures.to_dict()).

        Auto-adapts to the checkpoint's feature set (17-dim or 18-dim) by reading
        feature_names and feature_norms from the saved config.
        """
        ckpt_names = self.config.get("feature_names", [])
        ckpt_norms = self.config.get("feature_norms", {})
        if not ckpt_names:
            raise ValueError("Checkpoint has no feature_names metadata; "
                             "retrain with train_graph_prm.py or use get_reward() for legacy checkpoints")

        expected_dim = self.config.get("feature_dim", len(ckpt_names))
        if len(ckpt_names) != expected_dim:
            raise ValueError(f"feature_names length ({len(ckpt_names)}) != feature_dim ({expected_dim})")

        vec = []
        for name in ckpt_names:
            value = features_dict.get(name, 0)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            norm = ckpt_norms.get(name, 1.0)
            if norm == 0:
                norm = 1.0
            vec.append(float(value) / norm)

        with torch.no_grad():
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw = self.model(x).item()
            reward = 1 / (1 + np.exp(-raw)) if normalize else raw

        self.reward_history.append(reward)
        return reward

    def should_stop_from_dict(self, features_dict: dict) -> tuple:
        """
        Stopping decision from a raw feature dict (mirrors should_stop logic).

        Returns: (should_stop, reason, reward)
        """
        reward = self.get_reward_from_features_dict(features_dict, normalize=False)
        self.max_reward_seen = max(self.max_reward_seen, reward)
        current_step = len(self.reward_history)

        if current_step < self.min_steps:
            return False, "continue", reward

        total_ev = features_dict.get("total_evidence", features_dict.get("total_evidences", 0))
        has_evidence = total_ev > 0

        if not has_evidence:
            min_steps_no_ev = max(self.min_steps + 3, 6)
            if current_step < min_steps_no_ev:
                return False, "continue", reward
            return True, "no_evidence", reward

        if has_evidence and reward < self.max_reward_seen - self.decline_threshold:
            return True, "prm_decline", reward

        if has_evidence and total_ev >= 3 and current_step >= 4:
            recent = self.reward_history[-4:]
            if max(recent) - min(recent) < self.convergence_threshold:
                return True, "prm_converged", reward

        return False, "continue", reward

    def should_stop(self, features: StateFeatures) -> Tuple[bool, str, float]:
        """[LEGACY] Stopping decision from StateFeatures.

        Prefer should_stop_from_dict() with GraphFeatures.to_dict().
        Raises ValueError if checkpoint was trained on GraphFeatures.
        """
        reward = self.get_reward(features, normalize=False)

        # Update max reward seen
        self.max_reward_seen = max(self.max_reward_seen, reward)

        current_step = len(self.reward_history)

        # Don't stop too early - need minimum evidence
        if current_step < self.min_steps:
            return False, "continue", reward

        # Check if we have found any evidence at all
        has_evidence = features.total_evidences > 0

        # If no evidence found yet, be more patient - don't stop until we've tried more
        if not has_evidence:
            # Require more steps before giving up on a search with no evidence
            min_steps_no_evidence = max(self.min_steps + 3, 6)  # At least 6 steps
            if current_step < min_steps_no_evidence:
                return False, "continue", reward
            # Only stop after many fruitless steps
            if current_step >= min_steps_no_evidence:
                return True, "no_evidence", reward

        # Condition 1: Reward declined significantly from peak
        # This suggests we've passed the optimal stopping point
        # Only apply this if we have some evidence
        if has_evidence and reward < self.max_reward_seen - self.decline_threshold:
            return True, "prm_decline", reward

        # Condition 2: Reward converged (small change for multiple steps)
        # Check if reward changes have been consistently small
        # Only apply this if we have substantial evidence
        if has_evidence and features.total_evidences >= 3 and current_step >= 4:
            recent_rewards = self.reward_history[-4:]
            reward_range = max(recent_rewards) - min(recent_rewards)
            if reward_range < self.convergence_threshold:
                return True, "prm_converged", reward

        return False, "continue", reward

    def get_reward_trajectory(self) -> List[float]:
        """Get full reward history"""
        return self.reward_history.copy()


# Test function
if __name__ == "__main__":
    import sys

    # Test with GraphFeatures (active interface)
    model_path = "models/prm_v2_clean/graph_prm.pt"

    if Path(model_path).exists():
        print(f"Loading model from {model_path}")
        prm = PRM(model_path=model_path)
        print(f"Model config: feature_dim={prm.config.get('feature_dim')}, "
              f"feature_names={len(prm.config.get('feature_names', []))} features")

        test_dict = {
            "direct_path_count": 3, "direct_score_max": 1.2,
            "two_hop_count": 8, "two_hop_score_max": 0.6,
            "best_path_score": 1.2, "path_score_gap": 0.8,
            "contradiction_ratio": 0.15, "polarity_entropy": 0.7,
            "head_degree": 12, "tail_degree": 8,
            "common_neighbors": 3, "new_edges_ratio": 0.4,
            "path_discovery_rate": 0.3, "marginal_gain": 0.2,
            "total_evidence": 15, "avg_confidence": 0.75,
            "graph_density": 0.05, "recency_ratio": 0.6,
            "edge_uncertainty": 0.4, "constraint_violation": 0.1,
        }

        reward = prm.get_reward_from_features_dict(test_dict)
        print(f"Test reward: {reward:.4f}")

        should_stop, reason, _ = prm.should_stop_from_dict(test_dict)
        print(f"Should stop: {should_stop}, Reason: {reason}")
    else:
        print(f"Model not found at {model_path}")
