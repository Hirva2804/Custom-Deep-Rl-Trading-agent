"""
Custom PyTorch networks for the PPO actor-critic agent.

Bug #17 fix: ``set_training_mode`` is overridden in the policy to
correctly propagate train/eval mode to the custom network, ensuring
Dropout is disabled during inference.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Separate policy and value networks with LayerNorm + Dropout.

    Parameters
    ----------
    feature_dim : int
        Input feature dimensionality (from SB3 feature extractor).
    hidden_dim : int
        Width of the hidden layer.
    last_layer_dim_pi : int
        Output dim for the policy head.
    last_layer_dim_vf : int
        Output dim for the value head.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        dropout: float = 0.07,
    ) -> None:
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_layer_dim_pi),
            nn.LayerNorm(last_layer_dim_pi),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_layer_dim_vf),
            nn.LayerNorm(last_layer_dim_vf),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# ---------------------------------------------------------------------------
# Network factory — allows configs to control architecture
# ---------------------------------------------------------------------------

_NETWORK_REGISTRY: Dict[str, type] = {}


def _make_custom_network_class(
    hidden_dim: int = 256,
    output_dim: int = 64,
    dropout: float = 0.07,
) -> type:
    """
    Return a ``CustomNetwork`` subclass whose ``__init__`` is pre-configured
    with the given hyper-parameters.  This is needed because SB3 instantiates
    the MLP extractor with only ``feature_dim`` as argument.
    """
    key = (hidden_dim, output_dim, dropout)
    if key in _NETWORK_REGISTRY:
        return _NETWORK_REGISTRY[key]

    class _ConfiguredNetwork(CustomNetwork):
        def __init__(self, feature_dim: int) -> None:
            super().__init__(
                feature_dim,
                hidden_dim=hidden_dim,
                last_layer_dim_pi=output_dim,
                last_layer_dim_vf=output_dim,
                dropout=dropout,
            )

    _ConfiguredNetwork.__qualname__ = (
        f"CustomNetwork(h={hidden_dim},o={output_dim},d={dropout})"
    )
    _NETWORK_REGISTRY[key] = _ConfiguredNetwork
    return _ConfiguredNetwork


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    SB3-compatible actor-critic policy that uses :class:`CustomNetwork`.

    The class attribute ``_network_class`` must be set (via
    :func:`make_policy_class`) *before* instantiation so that
    ``_build_mlp_extractor`` injects the right architecture.
    """

    _network_class: type = CustomNetwork  # overridden by factory

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self._network_class(self.features_dim)

    # Bug #17 fix: propagate train/eval mode to custom sub-networks
    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


def make_policy_class(
    hidden_dim: int = 256,
    output_dim: int = 64,
    dropout: float = 0.07,
) -> type:
    """
    Create a ``CustomActorCriticPolicy`` subclass pre-configured with the
    given network hyper-parameters.

    Returns
    -------
    type
        A policy class suitable for passing to ``PPO(...)``.
    """
    net_cls = _make_custom_network_class(hidden_dim, output_dim, dropout)

    class _Policy(CustomActorCriticPolicy):
        _network_class = net_cls

    _Policy.__qualname__ = (
        f"CustomActorCriticPolicy(h={hidden_dim},o={output_dim},d={dropout})"
    )
    return _Policy
