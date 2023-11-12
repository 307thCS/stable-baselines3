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
from stable_baselines3.bit.policies import Actor, Critic, CnnPolicy, MlpPolicy, MultiInputPolicy, BITPolicy

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class BIT(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

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
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: BITPolicy
    actor: Actor
    actor_target: Actor
    critic: Critic
    critic_target: Critic

    def __init__(
        self,
        policy: Union[str, Type[BITPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        contrastive_loss_mult: int = 2,
        policy_update: int = 500,
        policy_temp: float = 1.,
        printout_interval: int = 50000,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        actor_delay: int = 2,
        grad_printout: bool = False,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = {"num_ideas": 8},
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
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
        self.grad_printout = grad_printout
        self.policy_delay, self.actor_delay = policy_delay, actor_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.idxrange = th.arange(batch_size).to(self.device)
        self.contrastive_loss_mult = contrastive_loss_mult
        self.num_ideas = policy_kwargs["num_ideas"]
        self.policy_update, self.policy_temp = policy_update, policy_temp
        self.printout_interval = printout_interval
        #self.inverse_eye = (1 - th.eye(self.num_ideas)).to(self.device)
        #self.tril = self.inverse_eye
        self.tril = th.ones(self.num_ideas, self.num_ideas).float().to(self.device).tril(diagonal=-1)
    
        self.start_points = th.ones(self.num_ideas)[None, :, None].float().to(self.device)
        
        self.actor_loss_grad_norm_ema = 0
        self.contrastive_loss_grad_norm_ema = 0
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
        self.policy_net = self.policy.policy
        self.policy_net_target = self.policy.policy_target
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.policy_net.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            if self._n_updates % (self.printout_interval * self.policy_delay) == 0:
                printout = True
            else:
                printout = False
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action_candidates = self.actor_target(replay_data.next_observations)
                candidate_values = self.critic_target.q1_forward(replay_data.next_observations, next_action_candidates, idx = 1).squeeze(dim=2)
                next_actions = next_action_candidates[self.idxrange, candidate_values.argmax(dim=1)]
                next_actions = (next_actions + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
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
            
            if self._n_updates % self.actor_delay == 0:
                # Compute actor loss
                actions = self.actor(replay_data.observations)
                base_values = self.critic.q1_forward(replay_data.observations, actions)
                values = -base_values.squeeze(dim=2)
                actor_loss = values.mean()
                actor_losses.append(actor_loss.item())
                if self.contrastive_loss_mult > 0:
                    contrastive_actor_loss = self.calculate_contrastive_loss(actions)
                    actor_losses.append(contrastive_actor_loss.item())
                    actor_loss += contrastive_actor_loss * self.contrastive_loss_mult
                # Optimize the actor
                actor_loss_grad_norm = 0
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_printout:
                    if self._n_updates % 40 == 0:
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                actor_loss_grad_norm += param.grad.data.norm(2).item()**2

                        actor_loss_grad_norm = (actor_loss_grad_norm ** 0.5) / len(list(self.actor.parameters()))
                        ema = 0.997
                        self.actor_loss_grad_norm_ema = self.actor_loss_grad_norm_ema * ema + actor_loss_grad_norm * (1 - ema)
                        if self._n_updates % 24000 == 0:    
                            print(f"Actor Loss Grad Norm: {self.actor_loss_grad_norm_ema}")
                self.actor.optimizer.step()
            if self._n_updates % self.policy_delay == 0:
                with th.no_grad():
                    if self._n_updates % self.actor_delay != 0:
                        actions = self.actor(replay_data.observations)
                    old_logits = self.policy_net_target(replay_data.observations, actions.detach())
                    base_values_2 = self.critic.q1_forward(replay_data.observations, actions, idx = 1).detach()
                    target_policy = self.adjust_policy(F.softmax(old_logits, dim=1), base_values_2.detach())
                base_logits = self.policy_net(replay_data.observations, actions.detach())
                policy_loss = self.log_loss(base_logits, target_policy.detach())
                regularization_loss = self.regularization_loss(base_logits)
                policy_loss = policy_loss + regularization_loss
                self.policy_net.optimizer.zero_grad()
                policy_loss.backward()
                self.policy_net.optimizer.step()
                
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
                if printout == True:
                    print("log loss:", policy_loss - regularization_loss)
                    print("regularization loss:", regularization_loss)
                    print("logits:", [round(i, 2) for i in old_logits[0].detach().squeeze().cpu().tolist()])
                    old_probs = F.softmax(old_logits, dim=1) 
                    old_policy_slice = old_probs[0].detach().squeeze().cpu()
                    prob_max = old_policy_slice.argmax()
                    self.print_percentages("old policy:   ", old_policy_slice)
                    target_policy_slice = target_policy[0].detach().squeeze().cpu()
                    self.print_percentages("target policy:", target_policy_slice)
                    adjustments = target_policy_slice - old_policy_slice
                    self.print_percentages("adjustments:  ", adjustments, digits=2)
                    a_values = base_values_2 - (base_values_2 * old_probs).sum(dim=1, keepdim=True)
                    value_max = a_values[0].argmax()
                    moves_and_values = th.cat((actions[0], a_values[0]), dim=1).detach().squeeze().cpu()
                    self.print_matrix("actions:", moves_and_values, digits=2)
                    print("highest prob and value index:", prob_max, value_max)
                    print("")
            if self._n_updates % self.policy_update == 0:
                self.policy_net_target.load_state_dict(self.policy_net.state_dict())
                
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BIT",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
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
    def calculate_contrastive_loss(self, actions):
        bs = actions.shape[0]
        distances = (abs(actions.unsqueeze(dim=1).detach() - actions.unsqueeze(dim=2))).mean(dim=3)
        losses = (th.nn.functional.relu(self.start_points - distances) * self.tril) ** 2 
        return losses.mean()
    def adjust_policy(self, old_probs, values):
        a_values = values - (values * old_probs).sum(dim=1, keepdim=True)
        adjustments = a_values * self.policy_temp
        new_probs = old_probs + adjustments
        new_probs = F.relu(new_probs)
        new_probs = new_probs / new_probs.sum(dim=1, keepdim=True)
        return new_probs
    def log_loss(self, pred_logits, targets):
        log_p = F.log_softmax(pred_logits, dim=1)
        loss = (-targets * log_p)
        loss = loss.sum(dim=1).mean()
        return loss
    def regularization_loss(self, logits):
        loss = (logits.mean() ** 2) / 100
        return loss
    def regularization_loss_2(self, logits):
        loss = (logits ** 2).mean() / 1000
        return loss
    def print_percentages(self, name, probs, digits=1):
        str_probs = [str(round(probs[i].tolist() * 100, digits)) + "%" for i in range(probs.shape[0])]
        print(name, str_probs)
    def print_matrix(self, name, numbers, digits=1):
        numbers_list = []
        for i in range(numbers.shape[0]):
            str_list = [str(round(numbers[i][j].tolist(), digits)) for j in range(numbers.shape[1])]
            for k in range(len(str_list)):
                if str_list[k][0] != '-':
                    str_list[k] = '+' + str_list[k]
                if len(str_list[k]) != 3 + digits:
                    str_list[k] = str_list[k] + '0'
                if k == len(str_list) - 1:
                    str_list[k] = 'advantage: ' + str_list[k]
            numbers_list.append(str_list)
        print(name)
        for i in range(len(numbers_list)):
            print(numbers_list[i])