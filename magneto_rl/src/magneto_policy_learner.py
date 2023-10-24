#!/usr/bin/env python3
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from typing import Tuple, Callable
# from stable_baselines3.common.callbacks import BaseCallback

# class saveWeights(BaseCallback):
#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)
    
#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass

#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.
#         """
#         pass

#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.

#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.

#         :return: (bool) If the callback returns False, training is aborted early.
#         """
#         return True

#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         pass

#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass

class ModelLearnerNetwork (torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        # Your policy_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_pi
        self.policy_net = torch.nn.Linear(feature_dim, last_layer_dim_pi)

        # Value network
        # Your value_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_vf
        self.value_net = torch.nn.Linear(feature_dim, last_layer_dim_vf)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ModelLearnerNetwork(self.features_dim)
