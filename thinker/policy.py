import json
import os
import random
import numpy as np
from typing import List, Dict, Any
from persistence import atomic_write_json

# ── Procedural Memory Policy ──────────────────────────────────────────────────

class CognitivePolicy:
    """
    Contextual Bandit for deciding which cognitive pattern to use.
    State: (node_type, cluster)
    Actions: cognitive_patterns
    """
    
    POLICY_PATH = "data/policy.json"
    
    DEFAULT_ACTIONS = [
        "analogical",
        "dialectical",
        "reductive",
        "experimental",
        "integrative",
    ]

    def __init__(self, epsilon: float = 0.2, learning_rate: float = 0.1,
                 softmax_temp: float = 0.35,
                 exploration_floor: float = 0.05,
                 uncertainty_bonus: float = 0.08):
        # Keep epsilon for backward compatibility of saved configs, but prefer
        # softmax-based selection to avoid hard branch heuristics.
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.softmax_temp = max(0.05, float(softmax_temp))
        self.exploration_floor = max(0.0, min(0.4, float(exploration_floor)))
        self.uncertainty_bonus = max(0.0, float(uncertainty_bonus))
        self.q_table: Dict[str, Dict[str, float]] = {}  # { "node_type|cluster": { "action": q_value } }
        self.action_counts: Dict[str, Dict[str, int]] = {}
        self._load()
    
    def _state_key(self, node_type: str, cluster: str) -> str:
        return f"{node_type}|{cluster}"
        
    def _init_state(self, state_key: str):
        if state_key not in self.q_table:
            # Initialize with slight optimism to encourage early exploration
            self.q_table[state_key] = {action: 0.5 for action in self.DEFAULT_ACTIONS}
        if state_key not in self.action_counts:
            self.action_counts[state_key] = {
                action: 0 for action in self.DEFAULT_ACTIONS
            }

    def _sanitize_state(self, state_key: str):
        self._init_state(state_key)
        prior = self.q_table.get(state_key, {})
        self.q_table[state_key] = {
            action: float(prior.get(action, 0.5))
            for action in self.DEFAULT_ACTIONS
        }
        prior_counts = self.action_counts.get(state_key, {})
        self.action_counts[state_key] = {
            action: int(prior_counts.get(action, 0))
            for action in self.DEFAULT_ACTIONS
        }

    def choose_pattern(self, node_type: str, cluster: str,
                       preferred_action: str = "") -> str:
        """Uncertainty-aware softmax action selection."""
        state_key = self._state_key(node_type, cluster)
        self._sanitize_state(state_key)
        actions = list(self.q_table[state_key].keys())
        preferred_action = (
            preferred_action if preferred_action in self.q_table[state_key] else ""
        )

        # Build logits from value + semantic preference + uncertainty bonus.
        logits = []
        counts = self.action_counts[state_key]
        pref_bonus = 0.10
        for action in actions:
            value = self.q_table[state_key][action]
            bonus = pref_bonus if action == preferred_action else 0.0
            uncertainty = self.uncertainty_bonus / np.sqrt(1.0 + counts[action])
            logit = (value + bonus + uncertainty) / self.softmax_temp
            logits.append(logit)

        # Stable softmax
        max_logit = max(logits)
        exp_vals = [np.exp(l - max_logit) for l in logits]
        total = sum(exp_vals)
        probs = [ev / total for ev in exp_vals]

        # Keep a small non-zero floor for all actions.
        if self.exploration_floor > 0.0:
            uniform = 1.0 / len(actions)
            probs = [
                ((1.0 - self.exploration_floor) * p) +
                (self.exploration_floor * uniform)
                for p in probs
            ]

        chosen = random.choices(actions, weights=probs, k=1)[0]
        chosen_prob = probs[actions.index(chosen)]
        pref_note = f", preferred={preferred_action}" if preferred_action else ""
        print(
            f"  [Procedural] Selected '{chosen}' "
            f"(p={chosen_prob:.2f}{pref_note}) for {state_key}"
        )
        return chosen

    def update(self, node_type: str, cluster: str, action: str, reward: float, dopamine: float = 0.5):
        """
        Update the expected value of an action in a state.
        Dopamine level modulates the learning rate.
        """
        state_key = self._state_key(node_type, cluster)
        self._sanitize_state(state_key)
        if action not in self.DEFAULT_ACTIONS:
            print(f"  [Procedural] Skipping unsupported pattern '{action}' for {state_key}")
            return
        
        old_val = self.q_table[state_key][action]
        self.action_counts[state_key][action] += 1
        
        # If dopamine is high, we learn faster from positive rewards.
        # If dopamine is low, we learn slower.
        adjusted_lr = self.learning_rate * (1.0 + dopamine)
        
        new_val = old_val + adjusted_lr * (reward - old_val)
        self.q_table[state_key][action] = new_val
        
        print(f"  [Procedural] Policy update: {state_key} + {action} -> rew={reward:.2f}, val: {old_val:.2f}->{new_val:.2f}")
        self._save()

    def _load(self):
        if os.path.exists(self.POLICY_PATH):
            try:
                with open(self.POLICY_PATH, 'r') as f:
                    raw = json.load(f)

                # Backward compatibility: historical files stored only q_table.
                if isinstance(raw, dict) and "q_table" in raw:
                    self.q_table = raw.get("q_table", {})
                    self.action_counts = raw.get("action_counts", {})
                else:
                    self.q_table = raw if isinstance(raw, dict) else {}
                    self.action_counts = {}

                for state_key in list(self.q_table.keys()):
                    self._sanitize_state(state_key)
                print(f"Loaded procedural policy with {len(self.q_table)} states.")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                self.q_table = {}
                self.action_counts = {}
                
    def _save(self):
        payload = {
            "q_table": self.q_table,
            "action_counts": self.action_counts,
            "meta": {
                "version": 2,
                "softmax_temp": self.softmax_temp,
                "exploration_floor": self.exploration_floor,
                "uncertainty_bonus": self.uncertainty_bonus,
            },
        }
        atomic_write_json(self.POLICY_PATH, payload)
