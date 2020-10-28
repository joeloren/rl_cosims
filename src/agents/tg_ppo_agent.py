import json
from collections import deque, defaultdict
from copy import deepcopy

import datetime
import gym
import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional

from src.models.tg_models import PolicyGNN
from src.training.torch_utils import get_available_device


class PPOAgent:
    def __init__(
            self,
            env: Optional[gym.Wrapper],
            config: dict,
            model: PolicyGNN,
            eval_seeds: List[int],
            baseline_eval_values: Dict[str, Dict[int, float]],
    ):
        self.config = config
        self.env: gym.Wrapper = env
        self.device = get_available_device()
        print(f'device chosen:{self.device}')
        self.policy: torch.nn.Module = model.to(self.device)
        self.old_policy = deepcopy(self.policy)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['lr'])
        self.episode_rewards = []
        self.minibatch_size = self.config['minibatch_size']
        self.n_ppo_updates = self.config['n_ppo_updates']
        self.target_kl = self.config['target_kl']
        # fpr clamped policy loss (PPO)
        self.eps_clip = self.config['eps_clip']
        self.logit_normalizer = self.config['logit_normalizer']
        self.episode_log_probabilities = []
        (
            self.state,
            self.reward,
            self.next_state,
            self.done,
            self.total_episode_score_so_far,
        ) = (None, None, None, None, None)
        self.episode_step_number = 0
        self.episode_number = 0
        self.episode_number_in_batch = 0
        self.reward_normalizer = 1
        # lr scheduler for changing the lr during training.
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.95,
            patience=200,
            cooldown=200,
            verbose=True,
        )
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        output_dir = f"runs/{self.config['problem_name']}/ppo_agent/{time_str}"
        path = Path(output_dir)
        if not path.exists():
            path.mkdir(parents=True)
        self.writer = SummaryWriter(log_dir=output_dir)
        self.config['model_save_dir'] = output_dir
        self.eval_seeds = eval_seeds
        self.evaluation_episodes = 0
        self.baseline_eval_values = baseline_eval_values or {}
        self.total_rewards_window = deque(
            maxlen=self.config['reward_average_window_size']
        )
        self.best_overall_models = []
        # dictionary where the key is the reward and the value is the episode number (used for
        # deleting the worst model saved so far)
        self.best_overall_model_episodes = {}
        self.batch_states, self.batch_rtgs, self.batch_actions, self.batch_log_probs, self.batch_vals, self.batch_advantage = [], [], [], [], [], []
        self.best_episode_score = np.inf

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.env.reset()
        self.next_state = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def train(self):
        # dump hyperparams to tb:
        self.writer.add_text(f'config/{self.config["run_name"]}', json.dumps(self.config, indent=True), 0)
        for episode in range(self.config['number_of_episodes']):
            self.train_episode()
            if episode % self.config['evaluate_every'] == 0 and episode > 0:
                print(f'Episode: {episode}, running evaluation')
                self.evaluate()
            if episode % self.config['save_checkpoint_every'] == 0 and episode > 0:
                print('Saving a checkpoint')
                self.save_checkpoint()

    def train_episode(self):
        """Runs a step within a game including a learning step if required"""
        # start the environment with a seed from a list of seeds
        self.env.seed(np.random.randint(0, self.config['num_train_seeds']))
        self.reset_game()
        loss = 0
        while not self.done:
            # print(f'calculating episode {self.episode_number},  step:{self.episode_step_number}')
            self.episode_step()
            # self.update_next_state_reward_done_and_score()
            # TODO need to update time_to_learn to be based on number of batches
            if self.time_to_learn():
                loss = self.learn()

            self.state = self.next_state  # this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1
        self.total_rewards_window.append(sum(self.episode_rewards))

    def episode_step(self, train=True):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it
        conducted to be used for
        learning later"""
        action, log_probability, val = self.select_action_and_log_prob(state=self.state)
        # store the state, action, reward and log-prob of this step (before action is preformed)
        self.next_state, self.reward, self.done, info = self.env.step(action)
        if train:
            self.store_step(state=self.state, reward=self.reward, action=action, log_prob=log_probability, val=val)
        if self.done:
            # add return to go (rtg) of this episode to batch rtgs
            self.batch_rtgs.extend(self.compute_episode_rtgs())
            if train:
                self.episode_number_in_batch += 1
        self.total_episode_score_so_far += self.reward

    def select_action_and_log_prob(self, state):
        """Picks actions and then calculates the log probabilities of the actions it picked given
        the policy"""
        action, logprob, value = self.policy.step(state, self.device)
        return action.item(), logprob.item(), value.item()

    def store_step(self, state, reward, action, log_prob, val):
        self.batch_states.append(state)
        if reward is not None:
            self.episode_rewards.append(self.reward)
        self.batch_actions.append(action)
        self.batch_log_probs.append(log_prob)
        self.batch_vals.append(val)

    def compute_episode_rtgs(self):
        """Calculates the cumulative discounted return for the episode"""
        discounted_rewards = []

        for t in range(len(self.episode_rewards)):
            rtg_t = 0
            for pw, r in enumerate(self.episode_rewards[t:]):
                # normalize rewards by a constant number
                rtg_t = rtg_t + self.config['discount'] ** pw * (r / self.reward_normalizer)
            discounted_rewards.append(rtg_t)
        return discounted_rewards

    def normalize_batch_advantage(self, batch_advantage_tensor):
        normalized_batch_advantage = (batch_advantage_tensor - batch_advantage_tensor.mean()) \
                                     / (batch_advantage_tensor.std() + 1e-9)
        return normalized_batch_advantage.to(dtype=torch.float32)

    def learn(self):
        """Runs a learning iteration for the policy"""
        total_num_steps = len(self.batch_rtgs)
        self.old_policy.load_state_dict(self.policy.state_dict())
        start_time = time.time()
        total_loss = 0
        for i in range(self.n_ppo_updates):
            self.optimizer.zero_grad()
            total_loss, chosen_logprobs_train, mean_kl = self.compute_loss()
            sliding_average_total_reward = -np.mean(self.total_rewards_window)
            total_loss.backward()
            self.optimizer.step()
            if mean_kl >= 1.5 * self.target_kl:
                print(f"Training: early stopping in PPO pass {i}. Reached approx. mean KL: {mean_kl}")
                break
        print(f'Total time for all PPO updates: {time.time() - start_time}s')
         # self.lr_scheduler.step(sliding_average_total_reward)
        # add things to tb and evaluate if needed -
        if (self.episode_number + 1) % 40 == 0:
            print(
                f'Finished episode {self.episode_number}, total loss:{total_loss:.2f},'
                f' ' f'total reward:{sum(self.episode_rewards)}')
            # log the network params histogram, used to understand if network is still learning
            for name, param in self.policy.named_parameters():
                self.writer.add_histogram('parameters/{}'.format(name), param.data, self.episode_number)
                self.writer.add_scalar(f'parameters/min-{name}', torch.min(param.grad), self.episode_number)
                self.writer.add_scalar(f'parameters/max-{name}', torch.max(param.grad), self.episode_number)
        self.writer.add_scalar('Train/Reward', sum(self.episode_rewards), self.episode_number)
        self.writer.add_scalar('Train/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.episode_number)
        self.writer.add_scalar('Train/AverageLogProb_data_collection', torch.mean(torch.tensor(self.batch_log_probs)),
                               self.episode_number)
        self.writer.add_scalar('Train/AverageLogProb_train', torch.mean(torch.tensor(chosen_logprobs_train)),
                               self.episode_number)
        self.writer.add_scalar('Train/MinRTGs', torch.min(torch.tensor(self.batch_rtgs)), self.episode_number)
        self.writer.add_scalar('Train/MaxRTGs', torch.max(torch.tensor(self.batch_rtgs)), self.episode_number)
        if torch.cuda.is_available():
            # log memory to see cuda information:
            mem_allocated = torch.cuda.memory_allocated(self.device)*1e-9
            mem_reserved = torch.cuda.memory_reserved(self.device)*1e-9
            self.writer.add_scalar('Memory/cuda_reserved', mem_reserved, self.episode_number)
            self.writer.add_scalar('Memory/cuda_allocated', mem_allocated, self.episode_number)

        self.reset_batches()
        return total_loss

    def reset_batches(self):
        self.batch_rtgs = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_log_probs = []
        self.batch_advantage = []
        self.batch_vals = []
        self.episode_number_in_batch = 0

    def compute_loss(self):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        original_logprobs = torch.tensor(self.batch_log_probs).to(device="cpu").view(-1, 1)
        batch_rtgs_tensor = torch.tensor(self.batch_rtgs, device="cpu", dtype=torch.float32).view(-1, 1)

        policy_results = self.policy.compute_probs_and_state_values(self.batch_states, self.batch_actions, self.device)
        chosen_logprob, batch_probabilities, batch_state_values = policy_results
        # add entropy for exploration
        # Policy loss
        batch_advantage_tensor = self.normalize_batch_advantage(batch_rtgs_tensor - batch_state_values.detach())
        ratio = torch.exp(chosen_logprob - original_logprobs)
        base_policy_loss = ratio * batch_advantage_tensor
        clamped_policy_loss = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantage_tensor
        policy_loss = -torch.min(base_policy_loss, clamped_policy_loss).mean()

        # normalized_batch_advantage = self.normalize_batch_advantage(batch_advantage_tensor)
        # Entropy loss
        entropy_loss = -batch_probabilities.mean() * self.config['entropy_coeff']
        value_loss = (F.mse_loss(batch_state_values, batch_rtgs_tensor) * self.config['value_coeff'])
        total_loss = policy_loss + entropy_loss + value_loss
        mean_kl = (original_logprobs - chosen_logprob).mean().detach().cpu().item()

        self.writer.add_scalar('Train/Loss/value', value_loss, self.episode_number)
        self.writer.add_scalar('Train/Loss/policy', policy_loss.mean(), self.episode_number)
        self.writer.add_scalar('Train/Loss/entropy', entropy_loss, self.episode_number)
        # self.writer.add_scalar('Train/Loss/total', total_loss, self.episode_number)
        return total_loss, chosen_logprob, mean_kl

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn
        at the end of every
        episode so this just returns whether the episode is over"""
        if self.done and self.episode_number_in_batch == self.config['number_of_episodes_in_batch']:
            return True
        else:
            return False

    def evaluate(self):
        seeds = np.random.choice(self.eval_seeds, self.config['num_eval_seeds'], replace=False)
        errors_dict = defaultdict(list)
        rewards_list = []

        for seed in seeds:
            total_rewards = 0
            self.env.seed(seed=seed)
            self.reset_game()
            while not self.done:
                self.episode_step(train=False)
                self.state = self.next_state  # this is to set the state for the next iteration
                self.episode_step_number += 1
                total_rewards += self.reward
            rewards_list.append(total_rewards)
            for baseline, values_dict in self.baseline_eval_values.items():
                relative_diff = (100 * (total_rewards - values_dict[seed]) / values_dict[seed])
                errors_dict[baseline].append(relative_diff)
        mean_reward = np.mean(rewards_list)
        mean_relative_diff = np.mean(list(errors_dict.values())[0])
        self.writer.add_scalar('Eval/Reward per eval', mean_reward, self.evaluation_episodes)
        self.writer.add_scalar('Eval/Reward per episode', mean_reward, self.episode_number)
        # save the best overall model -
        if len(self.best_overall_models) >= 1:
            print( f"current relative:{mean_relative_diff},"
                   f" max relative:{max(self.best_overall_models, key=lambda x: x[0])[0]}")
        if (len(self.best_overall_models) < 3 or
                mean_relative_diff < max(self.best_overall_models, key=lambda x: x[0])[0]):
            self.save_model()
            self.best_overall_models.append((mean_relative_diff, self.episode_number))

            # remove the lowest-scoring among the top models
            if len(self.best_overall_models) > 3:
                max_score_index = self.best_overall_models.index(max(self.best_overall_models, key=lambda x: x[0]))
                reward_del, episode_number_to_delete = self.best_overall_models.pop(max_score_index)
                self.delete_model(episode_number_to_delete)
            top_models_df = (pd.DataFrame(data=self.best_overall_models, columns=['mean_relative_diff', 'episode_num'])
                             .sort_values(by='mean_relative_diff', ascending=False))
            top_models_df.to_csv(Path(self.config['model_save_dir']) / 'top_models.tsv', sep='\t', index=False)
        # print(f'Done evaluating on {len(seeds)}, resulting relative diffs: {errors_dict}')
        for baseline, relative_diffs in errors_dict.items():
            mean_relative_diff = np.mean(relative_diffs)
            print(f'Eval/Relative diff - {baseline} : {mean_relative_diff}')
            self.writer.add_scalar(f'Eval/Relative diff per eval- {baseline}', mean_relative_diff,
                                   self.evaluation_episodes)
            self.writer.add_scalar(f'Eval/Relative diff per episode - {baseline}', mean_relative_diff,
                                   self.episode_number)
        self.evaluation_episodes += 1

    def save_model(self):
        # print(f'saving model, episode {self.episode_number}')
        dir_name = self.config['model_save_dir']
        file_name = f'model_ep_{self.episode_number}'
        torch.save(self.policy.state_dict(), Path(dir_name) / file_name)
        conf_file = Path(dir_name) / 'agent_params.json'
        with conf_file.open(mode='w') as f:
            json.dump(obj=self.config, fp=f, indent=2)

    def delete_model(self, episode_num):
        dir_name = self.config['model_save_dir']
        file_name = f'model_ep_{episode_num}'
        file_path = Path(dir_name) / file_name
        if file_path.exists():
            file_path.unlink()
        # os.remove(os.path.join(dir_name, file_name))

    def get_checkpoint(self):
        checkpoint = {
            'episode_number': self.episode_number,
            'old_policy': self.old_policy.state_dict(),
            'policy': self.policy.state_dict(),
            'config': self.config,
            'optimizer': self.optimizer.state_dict(),
            'batch_rtgs': self.batch_rtgs,
            'batch_states': self.batch_states,
            'batch_logprobs': self.batch_log_probs,
            'batch_actions': self.batch_actions,
            'eval_seeds': self.eval_seeds,
            'baseline_eval_values': self.baseline_eval_values
        }
        return checkpoint

    def save_checkpoint(self, filename='./checkpoint.pth.tar'):
        checkpoint = self.get_checkpoint()
        checkpoint_path = Path(self.config['model_save_dir']) / filename

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path: str):
        """
        Loads the agent with the checkpoint information
        Note: the environment is set to None
        :param checkpoint_path:
        :return:
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model: PolicyGNN = PolicyGNN(cfg=checkpoint['config']['model_config'], model_name='ppo_policy_model')
        model.load_state_dict(checkpoint['policy'])
        old_model = deepcopy(model)
        old_model.load_state_dict(checkpoint['old_policy'])
        agent = PPOAgent(None, checkpoint['config'], model, checkpoint['eval_seeds'],
                         checkpoint['baseline_eval_values'])
        agent.old_policy = old_model
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.episode_number = checkpoint['episode_number']
        agent.batch_states = checkpoint['batch_states']
        agent.batch_rtgs = checkpoint['batch_rtgs']
        agent.batch_actions = checkpoint['batch_actions']
        agent.batch_log_probs = checkpoint['batch_logprobs']
        return agent
