import numpy as np
import gym
from gym import spaces


class LotSizingEnv:

    def __init__(self, total_timesteps: int, demands: list, capacities: list,
                holding_cost: float, setup_cost: float, starting_inventory: float, shortage_penalty: float):

        self.action_space = spaces.Box(low=0, high=1, shape=(2,))
        self.num_actions = 2

        self._demands = demands
        self._capacities = capacities
        self._holding_cost = holding_cost
        self._setup_cost = setup_cost
        self._shortage_penalty = shortage_penalty

        self.total_timesteps = total_timesteps

        self._timestep = 0
        self.last_reward = None
        self.last_action = None
        self.starting_inventory = starting_inventory
        self._prev_inventory = starting_inventory
        self.inventory_level = [starting_inventory]

    def step(self, action):
        
        reward = self._act(action)
        done = True if (self._timestep+1 == self.total_timesteps) else False
        if not done:
            new_observation = self._get_obs()
            self._timestep += 1
        else:
            self.reset()
            new_observation = self._get_obs()
        # self.is_done(new_observation)
        
        new_state = self._obs_to_state(new_observation)
        return(new_state, reward, done)

    def _act(self, action):
        cap = self._capacities[self._timestep]
        demand = self._demands[self._timestep]
        produce_flag = 1 if action[0] >= 0.5 else 0
        produce = action[1] * cap if produce_flag == 1 else 0

        new_inventory = self._prev_inventory + produce - demand

        reward = 0
        if new_inventory >= 0:
            reward += (-new_inventory*self._holding_cost)
        else:
            reward += (new_inventory*self._shortage_penalty)
        
        if produce_flag == 1:
            reward -= self._setup_cost
        
        self.last_reward = reward
        self.last_action = action 
        self.inventory_level.append(new_inventory)
        self._prev_inventory = new_inventory

        return(reward)

    # def is_done(self, obs: dict):
    #     if obs.get("timestep") == self.total_timesteps:
    #         return(True)
    #     else:
    #         return(False)

    def reset(self):
        self._timestep = 0
        self.last_reward = None
        self.last_action = None
        self._prev_inventory = self.starting_inventory

        observation = self._get_obs()
        state = self._obs_to_state(observation)

        return(state)
    
    def _obs_to_state(self, observation: dict):
        state = []
        for el in observation.values():
            state += el
        return(np.array(state))

    def _get_obs(self):
        return({
            "demands": self._demands,
            "setup_cost": [self._setup_cost],
            "holding_cost": [self._holding_cost],
            "prev_inventory": [self._prev_inventory],
            "timestep": [self._timestep]
            })

    # def render(self, mode='human'):
    #     pass
    # def close (self):
    #     pass