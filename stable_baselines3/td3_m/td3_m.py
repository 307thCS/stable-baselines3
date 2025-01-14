from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3_m.policies import Actor, Critic, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3MPolicy

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3M(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG + Multiheaded Actor (TD3-M)
    
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param contrastive_loss_mult: temperature for the auxiliary contrastive loss
    :param start_epsilon: initial value of epsilon for epsilon-greedy exploration algorithm,
    :param min_epsilon: minimum value of epsilon,
    :param min_epsilon_by: fraction of training that epsilon reaches min epsilon value. 0.4 would mean it reaches min value 40% of way through training.
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3MPolicy
    actor: Actor
    actor_target: Actor
    critic: Critic
    critic_target: Critic

    def __init__(
        self,
        policy: Union[str, Type[TD3MPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = {"num_ideas": 8},
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        contrastive_loss_mult: int = 1,
        start_epsilon: float = 1.,
        min_epsilon: float = 0.,
        min_epsilon_by: float = 0.5,
        argmax_fraction: float = 0.5,
        cl_type: int = 2,
        target_type: int = 1,
        decrement: float = 0.25,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )
        self.argmax_fraction = argmax_fraction
        self.cl_type, self.target_type = cl_type, target_type
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.idxrange = th.arange(batch_size).to(self.device)
        self.contrastive_loss_mult = contrastive_loss_mult
        self.num_ideas = policy_kwargs["num_ideas"]
        minimum = 1 - decrement * (self.num_ideas - 1)
        self.rank_weights = th.relu(th.linspace(minimum, 1, self.num_ideas)[None, :].expand(batch_size, self.num_ideas)).to(self.device)
        self.rank_weights = self.rank_weights / self.rank_weights.sum(dim=1, keepdim=True)
        self.start_epsilon, self.min_epsilon, self.min_epsilon_by = start_epsilon, min_epsilon, min_epsilon_by
        self.tril = th.ones(self.num_ideas, self.num_ideas).float().to(self.device).tril(diagonal=-1)
        self.start_points = th.ones(self.num_ideas)[None, :, None].float().to(self.device)
        if _init_setup_model:
            self._setup_model()
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.policy.epsilon > self.min_epsilon:
            self.policy.epsilon -= self.epsilon_decrement * gradient_steps
            if self.policy.epsilon < self.min_epsilon:
                self.policy.epsilon = self.min_epsilon
        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            with th.no_grad():
                # Select action according to policy and add clipped noise
                next_action_candidates = self.actor_target(replay_data.next_observations)
                
                if self.target_type == 1:
                    candidate_values = self.critic_target.q1_forward(replay_data.next_observations, next_action_candidates, idx = 1).squeeze(dim=2)
                    max_indexes = candidate_values.argmax(dim=1)
                    next_actions = next_action_candidates[self.idxrange, max_indexes]
                    noise = next_actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (next_actions + noise).clamp(-1, 1)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                elif self.target_type == 2:
                    candidate_values = self.critic_target.q1_forward(replay_data.next_observations, next_action_candidates, idx = 1).squeeze(dim=2)
                    max_indexes = candidate_values.argmax(dim=1)
                    noise = next_action_candidates.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_action_candidates = (next_action_candidates + noise).clamp(-1, 1)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_action_candidates), dim=2)
                    next_q_values, _ = th.min(next_q_values, dim=2, keepdim=True)
                    next_q_values = next_q_values[self.idxrange, max_indexes] * self.argmax_fraction + next_q_values.mean(dim=1) * (1 - self.argmax_fraction)
                
                elif self.target_type == 3:
                    candidate_values = self.critic_target(replay_data.next_observations, next_action_candidates)
                    #print(candidate_values[0].shape, self.critic_target.q1_forward(replay_data.next_observations, next_action_candidates, idx = 1).shape)
                    max_indexes_1, max_indexes_2 = candidate_values[0].squeeze(dim=2).argmax(dim=1), candidate_values[1].squeeze(dim=2).argmax(dim=1)
                    next_actions = th.stack((next_action_candidates[self.idxrange, max_indexes_1], next_action_candidates[self.idxrange, max_indexes_2]), dim=1)
                    noise = next_actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (next_actions + noise).clamp(-1, 1)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=2)
                    next_q_values, _ = th.min(next_q_values, dim=2, keepdim=True)
                    next_q_values = next_q_values.mean(dim=1)
                elif self.target_type == 4:
                    candidate_values = self.critic_target.q1_forward(replay_data.next_observations, next_action_candidates, idx = 1).squeeze(dim=2)
                    idxs = candidate_values.argsort(dim=1).to(self.device)
                    weight_tensor = th.empty_like(self.rank_weights).to(self.device)
                    weight_tensor = weight_tensor.scatter_(1, idxs, self.rank_weights).unsqueeze(dim=2)
                    noise = next_action_candidates.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_action_candidates = (next_action_candidates + noise).clamp(-1, 1)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_action_candidates), dim=2)
                    next_q_values, _ = th.min(next_q_values, dim=2, keepdim=True)
                    next_q_values = (next_q_values * weight_tensor).sum(dim=1)
                    '''if self._n_updates == 1:
                        print("candidate_values:", candidate_values[0])
                        print("idxs:", idxs[0])
                        print("weight_tensor:", weight_tensor[0])
                        print("noise:", noise[0])
                        print("next_action_candidates:", next_action_candidates[0])
                        print("next_q_values:", next_q_values[0])'''
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actions = self.actor(replay_data.observations)
                #actions.retain_grad()
                values = -self.critic.q1_forward(replay_data.observations, actions).squeeze(dim=2)#-actions * 0.002
                actor_loss = values.mean()
                actor_losses.append(actor_loss.item())
                if self.contrastive_loss_mult > 0:
                    if self.cl_type == 1:
                        contrastive_actor_loss = self.calculate_contrastive_loss_1(actions)
                    elif self.cl_type == 2:
                        contrastive_actor_loss = self.calculate_contrastive_loss_2(actions)
                    actor_losses.append(contrastive_actor_loss.item())
                    actor_loss += contrastive_actor_loss * self.contrastive_loss_mult
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
                #if self._n_updates % 10000 == 0:
                #    print(actions.mean(dim=0).detach().cpu())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3-M",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        self.policy.epsilon = self.start_epsilon
        self.epsilon_decrement = (self.start_epsilon - self.min_epsilon) / ((total_timesteps - self.learning_starts) * self.min_epsilon_by)
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
    def calculate_contrastive_loss_1(self, actions):
        bs = actions.shape[0]
        distances = (abs(actions.unsqueeze(dim=2) - actions.unsqueeze(dim=1).detach())).mean(dim=3)
        losses = (th.nn.functional.relu(self.start_points - distances) * self.tril) ** 2 
        return losses.mean()
    def calculate_contrastive_loss_2(self, actions):
        bs = actions.shape[0]
        distances = ((actions.unsqueeze(dim=2) - actions.unsqueeze(dim=1).detach()) ** 2 + 1e-7).mean(dim=3) ** 0.5
        losses = (th.nn.functional.relu(self.start_points - distances) * self.tril) ** 2 
        return losses.mean()
    def target_adjust(values, rank_weights):
        idxs = values.argsort(dim=1)
        rank_tensor = th.empty_like(rank_weights).to(values.device).detach()
        rank_tensor.scatter_(1, idxs, rank_weights)
        values = (values * rank_tensor)
        return values