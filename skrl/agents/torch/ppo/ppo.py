from __future__ import annotations

import os
from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .ppo_cfg import PPO_CFG

from skrl.nn_models.lstm_uncertainty_estimator import LSTM_Unc
import torch.optim as optim

def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param truncated: Signals to indicate that episodes have been truncated.
    :param values: Values obtained by the agent.
    :param last_values: Last values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.
    :param time_limit_bootstrap: Whether to use time-limit (truncation) bootstrapping.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def likelihood_loss(mean_pred, mean_true, ln_sig_sq):
    return (1/2)*torch.mean(torch.sum(torch.exp(-1*ln_sig_sq)*torch.square(mean_true-mean_pred), dim=1) + torch.sum(ln_sig_sq, dim=1))

def normalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def denormalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - new_min) / (new_max - new_min) * (max_val - min_val) + min_val


class PPO(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: PPO_CFG | dict = {},
    ) -> None:
        """Proximal Policy Optimization (PPO).

        https://arxiv.org/abs/1707.06347

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: PPO_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=PPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # - optimizers
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self.cfg.learning_rate[0]
                )
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate schedulers
            self.scheduler = self.cfg.learning_rate_scheduler[0]
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler[0](
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_values = None
        self._rollout = 0

        # Initialise the 5 prop estimators for properties and their uncertainty estimation
        self.prop_models = []
        input_size = 4 
        hidden_size = 64
        num_layers = 1
        output_size = 3
        self.num_epochs = 5
        self.all_prop_names = ["static_friction", "dynamic_friction", "restitution"]
        for i in range(5):
            self.prop_models.append(LSTM_Unc(input_size, hidden_size, num_layers, output_size, i).to(self.device))
        self.prop_criterion = likelihood_loss
        self.mse = nn.MSELoss()
        prop_learning_rates = [0.01, 0.003, 0.001, 0.0003, 0.0001]
        self.prop_optimizers = [optim.Adam(model.parameters(), lr=prop_learning_rates[i]) for i, model in enumerate(self.prop_models)]
        print("Prop estimator models initialized.")
        # print(self.prop_models)

        # Load trained property estimator models in eval 
        # TODO: Fix hardcoding of path and training mode 
        trained_prop_estimators = True 
        if trained_prop_estimators: 
            pre_trained_path_dir = "/workspace/sliding/logs/skrl/sliding_newnew/2026-06-18_18-57-58_ppo_torch/checkpoints_prop/"
            for i in range(5): 
                curr_lstm_path = "LSTM_" + str(i) + "_best.pth"
                curr_pre_trained_path = os.path.join(pre_trained_path_dir, curr_lstm_path)
                print(curr_pre_trained_path)
                self.prop_models[i].load_state_dict(torch.load(curr_pre_trained_path, map_location=torch.device(self.device)))

            print("Pre-trained model loaded")
        else:
            print("Training property estimator models from scratch.")

        # Initialise property estimator observation and target buffer 
        self.normalised_curr_rollout_lstm_input = []
        self.normalised_curr_rollout_lstm_target = []

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, infos: dict[str, Any] | None = None, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """

        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

            # compute values
            if self.training:
                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)

        # print("Actions from policy model:", actions.shape)
        # print("Outputs from policy model:", type(outputs), outputs.keys())

        if infos: 
            # Get obs for property estimator 
            normalised_curr_lstm_prop_input = infos["prop_estimator_obs"]
            normalised_curr_lstm_prop_input = normalised_curr_lstm_prop_input.clone()
            self.normalised_curr_rollout_lstm_input.append(normalised_curr_lstm_prop_input.detach().clone())

            # Get target property values for property estimator training 
            normalised_curr_lstm_prop_target = infos["target_prop_values"]
            self.normalised_curr_rollout_lstm_target.append(normalised_curr_lstm_prop_target.detach().clone())

            estimates = []
            ln_sig_sqs = []
            with torch.no_grad():
                for model in self.prop_models:
                    estimate, ln_sig_sq = model(normalised_curr_lstm_prop_input)
                    estimates.append(estimate)
                    ln_sig_sqs.append(ln_sig_sq)

            mean_normalized_estimates = torch.stack(estimates, dim=1).mean(dim=1)
            mean_normalized_sig_sq = torch.exp(torch.stack(ln_sig_sqs, dim=1).mean(dim=1))

            self.prop_normalisation_dict = infos["prop_normalisation_dict"]

            denormalsied_output_list = []
            denormalsied_target_list = []

            for i, curr_prop_name in enumerate(self.all_prop_names):

                prop_min, prop_max = self.prop_normalisation_dict[curr_prop_name]
                tgt_min, tgt_max = self.prop_normalisation_dict["estimate_target"]

                denormalsied_output_list.append(
                    denormalize(
                        mean_normalized_estimates[:, i].reshape(-1, 1),
                        prop_min,
                        prop_max,
                        tgt_min,
                        tgt_max,
                    )
                )

                denormalsied_target_list.append(
                    denormalize(
                        normalised_curr_lstm_prop_target[:, i].reshape(-1, 1),
                        prop_min,
                        prop_max,
                        tgt_min,
                        tgt_max,
                    )
                )

            denormalsied_output = torch.cat(denormalsied_output_list, dim=1)
            denormalsied_target = torch.cat(denormalsied_target_list, dim=1)

            rnn_losses = []
            for estimate, ln_sig_sq in zip(estimates, ln_sig_sqs):
                rnn_losses.append(self.prop_criterion(estimate, normalised_curr_lstm_prop_target, ln_sig_sq))

            rnn_rmse = torch.sqrt(self.mse(denormalsied_output, denormalsied_target))

            squared_error = (denormalsied_output - denormalsied_target) ** 2
            rnn_rmse_per_prop = torch.sqrt(
                squared_error.mean(dim=0)
            )

            # This gives [num_envs, num_prop] for reward computation per env 
            rnn_normalised_abserr_env_prop = torch.abs(mean_normalized_estimates - normalised_curr_lstm_prop_target)

            mean_squares = torch.stack([torch.square(estimate) for estimate in estimates], dim=1).mean(dim=1)
            square_mean = torch.square(mean_normalized_estimates)
            epistemic_uncertainty_normalized = mean_squares - square_mean
            total_uncertainty_normalized = mean_normalized_sig_sq + epistemic_uncertainty_normalized

            prop_estimator_output = {
                "rnn_loss": sum(rnn_losses)/len(rnn_losses), 
                "rnn_rmse": rnn_rmse, 
                "rnn_rmse_staticfric": rnn_rmse_per_prop[0], 
                "rnn_rmse_dynamicfric": rnn_rmse_per_prop[1],
                "rnn_rmse_restitution": rnn_rmse_per_prop[2],
                "rnn_normalised_abserr_env_staticfric_env": rnn_normalised_abserr_env_prop[:, 0],
                "rnn_normalised_abserr_env_dynamicfric_env": rnn_normalised_abserr_env_prop[:, 1],
                "rnn_normalised_abserr_env_restitution_env": rnn_normalised_abserr_env_prop[:, 2],
                "normalized_output": mean_normalized_estimates, 
                "denormalsied_output": denormalsied_output, 
                "denormalsied_target": denormalsied_target,
                "aleatoric_uncertainty_normalized" : mean_normalized_sig_sq,
                "epistemic_uncertainty_normalized" : epistemic_uncertainty_normalized,
                "total_uncertainty_normalized" : total_uncertainty_normalized
            }

        # print(actions.shape)
        # print(outputs.keys())
        outputs["prop_estimator_output"] = prop_estimator_output

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap and truncated.any():
                with torch.no_grad():
                    inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    next_values, _ = self.value.act(inputs, role="value")
                    next_values = self._value_preprocessor(next_values, inverse=True)

                rewards += self.cfg.discount_factor * next_values * truncated

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

                    # Property estimator update
                    for model in self.prop_models:
                        model.train()
                    self.update_prop_estimator(timestep=timestep, timesteps=timesteps)
                    for model in self.prop_models:
                        model.eval()

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            import os
            log_model_dir = os.path.join(self.experiment_dir, "checkpoints_prop")
            if not os.path.exists(log_model_dir):
                os.makedirs(log_model_dir)

            for i, model in enumerate(self.prop_models):
                best_model_path = log_model_dir + f"/LSTM_{i}_best.pth"
                torch.save(model.to(self.device).state_dict(), best_model_path)

                curr_model_path = log_model_dir + f"/LSTM_{i}_"+str(timestep)+".pth"
                torch.save(model.to(self.device).state_dict(), curr_model_path)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = self.value.act(inputs, role="value")

                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self.cfg.grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / (self.cfg.learning_epochs * self.cfg.mini_batches))
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])


    def update_prop_estimator(self, timestep: int, timesteps: int) -> None:        
        for epoch in range(self.num_epochs):
            for i, rnn_input in enumerate(self.normalised_curr_rollout_lstm_input):
                for model, optimizer in zip(self.prop_models, self.prop_optimizers):
                    # print("RNN input shapeeeeeeeeeeeeeeeeeeee")
                    # print(rnn_input.shape)
                    # print(i)
                    # print(self.normalised_curr_rollout_lstm_target[i].shape)

                    targets = self.normalised_curr_rollout_lstm_target[i]
                    
                    outputs, ln_sig_sq = model(rnn_input)

                    # mean_friction = 0.5
                    # weight = 1+ torch.abs(targets-mean_friction)
                    # loss = torch.mean(weight*(outputs-targets)**2)
                    
                    # Assuming weights is a 1D tensor with 4 elements (one for each feature)
                    # weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=outputs.device)  # Replace w1, w2, w3, w4 with your weights
                    # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=outputs.device)  # Replace w1, w2, w3, w4 with your weights

                    # # Compute the element-wise loss
                    # self.prop_criterion_noreduction = nn.MSELoss(reduction='none')
                    # elementwise_loss = self.prop_criterion_noreduction(outputs, targets)  # Ensure no reduction yet

                    # # Scale each feature's loss by its respective weight
                    # weighted_loss = elementwise_loss * weights

                    # Reduce to a scalar loss (e.g., mean across all samples and features)
                    # loss = weighted_loss.mean()
                    loss = self.prop_criterion(outputs, targets, ln_sig_sq)
                                    
                    # loss = self.prop_criterion(outputs, targets)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # # Record average loss after each epoch to weights and biases
            # wandb.log({"loss": loss.item()})



        self.normalised_curr_rollout_lstm_input = []
        self.normalised_curr_rollout_lstm_target = []

        # rewards = self.memory.get_tensor_by_name("rewards")
        # states = self.memory.get_tensor_by_name("states")

        # print(rewards.shape)
        # print(states.shape)