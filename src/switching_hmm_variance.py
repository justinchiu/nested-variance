"""
Toy switching-HMM to compare gradient estimators for MoE routing.

We model a small HMM with K experts (routes). Each expert k supplies its own
transition matrix P^{(k)} over discrete states. A router rho_phi(z_t | s_t)
selects the expert at each timestep. We consider:

- Routing-replay gradient: sum_t ∇_phi log rho(z_t | s_t) (complete data).
- Marginal gradient: ∇_phi log sum_k rho(k|s_t) * P^{(k)}(s_t, s_{t+1}).
- Single-sample REINFORCE estimator for the marginal gradient:
    E_{k ~ rho}[ (P^{(k)}(s_t, s_{t+1}) / mix_prob) * ∇ log rho(k|s_t) ].
  This matches the nested-MC structure and can have high variance.

We simulate episodes and report empirical variances of the routing-replay and
single-sample marginal estimators (flattened gradients) as a function of horizon T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def jacobian_softmax(probs: np.ndarray) -> np.ndarray:
    # J_{ij} = d rho_i / d logits_j
    diag = np.diag(probs)
    outer = np.outer(probs, probs)
    return diag - outer


@dataclass
class SwitchingHMM:
    P: np.ndarray  # shape (K, S, S)
    phi: np.ndarray  # router logits, shape (S, K)
    S: int
    K: int

    @classmethod
    def default(cls) -> "SwitchingHMM":
        # Simple 3-state, 2-expert example.
        S = 3
        K = 2
        # Expert 0: mostly stays put.
        P0 = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        # Expert 1: tends to move toward state 2.
        P1 = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.1, 0.8],
            ]
        )
        P = np.stack([P0, P1], axis=0)
        phi = np.zeros((S, K))  # start with uniform router
        return cls(P=P, phi=phi, S=S, K=K)

    def sample_episode(self, T: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Sample states and routes for one episode."""
        states = np.empty(T + 1, dtype=np.int64)
        routes = np.empty(T, dtype=np.int64)
        states[0] = rng.integers(self.S)
        for t in range(T):
            probs = softmax(self.phi[states[t]])
            routes[t] = rng.choice(self.K, p=probs)
            next_probs = self.P[routes[t], states[t]]
            states[t + 1] = rng.choice(self.S, p=next_probs)
        return states, routes

    def grad_routing_replay(self, states: np.ndarray, routes: np.ndarray) -> np.ndarray:
        """Complete-data gradient sum_t ∇ log rho(z_t | s_t) (no reward scaling)."""
        grad = np.zeros_like(self.phi)
        for s, z in zip(states[:-1], routes):
            probs = softmax(self.phi[s])
            grad[s] += one_hot(z, self.K) - probs
        return grad

    def grad_marginal_exact(self, states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
        """
        Exact gradient of log mixture transition:
          log mix_t = log sum_k rho(k|s_t) * P^{(k)}(s_t, s_{t+1})
        """
        grad = np.zeros_like(self.phi)
        for s, sp in zip(states[:-1], next_states[1:]):
            probs = softmax(self.phi[s])  # (K,)
            trans_vec = self.P[:, s, sp]  # (K,)
            mix = float(np.dot(probs, trans_vec))
            J = jacobian_softmax(probs)  # (K,K)
            grad[s] += J.T @ trans_vec / mix
        return grad

    def grad_marginal_single_sample(
        self, states: np.ndarray, next_states: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Single-sample REINFORCE estimator for grad log mixture:
          E_{k ~ rho} [ (trans_k / mix) * ∇ log rho(k|s_t) ]
        This is an unbiased estimator of the exact marginal gradient, but can have high variance.
        """
        grad = np.zeros_like(self.phi)
        for s, sp in zip(states[:-1], next_states[1:]):
            probs = softmax(self.phi[s])
            z = rng.choice(self.K, p=probs)
            trans_vec = self.P[:, s, sp]
            mix = float(np.dot(probs, trans_vec))
            weight = trans_vec[z] / mix
            grad[s] += weight * (one_hot(z, self.K) - probs)
        return grad


def one_hot(i: int, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=float)
    v[i] = 1.0
    return v


def run_variance_study(
    T_values: List[int],
    episodes: int = 5000,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    hmm = SwitchingHMM.default()

    for T in T_values:
        grads_rr = []
        grads_marg_mc = []
        grads_marg_exact = []
        for _ in range(episodes):
            states, routes = hmm.sample_episode(T=T, rng=rng)
            grad_rr = hmm.grad_routing_replay(states, routes)
            grad_exact = hmm.grad_marginal_exact(states, states)
            grad_mc = hmm.grad_marginal_single_sample(states, states, rng)
            grads_rr.append(grad_rr.reshape(-1))
            grads_marg_exact.append(grad_exact.reshape(-1))
            grads_marg_mc.append(grad_mc.reshape(-1))

        grads_rr = np.stack(grads_rr)
        grads_marg_mc = np.stack(grads_marg_mc)
        grads_marg_exact = np.stack(grads_marg_exact)

        # Compute per-component variances and report mean variance across components.
        var_rr = grads_rr.var(axis=0).mean()
        var_mc = grads_marg_mc.var(axis=0).mean()
        # Norm of exact gradient to show scale.
        mean_norm_exact = np.linalg.norm(grads_marg_exact.mean(axis=0))

        print(f"T={T:2d} | mean var RR: {var_rr:.4f} | mean var marginal-1samp: {var_mc:.4f} | ||E[grad_exact]||={mean_norm_exact:.4f}")


if __name__ == "__main__":
    run_variance_study(T_values=[5, 10, 20], episodes=10000, seed=42)
