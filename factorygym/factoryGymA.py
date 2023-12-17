import copy
import random
from typing import Tuple, Optional, Union, List

import numpy as np

import pandas as pd
import gym
from gym.spaces import MultiBinary, Discrete, MultiDiscrete, Box
from gym.core import ActType, ObsType, RenderFrame
from factory.factorySim import FactorySim
import plotly.express as px


class ObservationSpace:
    def __init__(self,agents):
        self.queue_state = MultiBinary(2)
        self.action_state = Discrete(3)
        self.current_job_state = MultiDiscrete(
            [np.iinfo(np.int32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max])
        self.energy_state = gym.spaces.Tuple((Discrete(np.iinfo(np.int32).max),
                                              Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                                              Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)))
        self.observation_space = gym.spaces.Dict({'queue_state': self.queue_state,
                                                  'action_state': self.action_state,
                                                  'current_job_state': self.current_job_state})
        obs_space = gym.spaces.Dict({})
        for i in range(agents):
            agent_obs_space = self.observation_space
            obs_space.spaces[f"agent{i}"] = agent_obs_space



class FsimEnv(gym.Env):

    def __init__(self, num_products, num_machines, products_xml, machines_xml, energy_xml):

        self.record = []
        self.factory_sim = FactorySim()
        self.factory_sim.create_products(products_xml, num_products)
        self.factory_sim.create_machines(machines_xml, num_machines)
        self.factory_sim.create_energy_market(energy_xml)
        self.factory_sim.create_jobs()
        for machine in self.factory_sim.machine_simulator.manpy_machine_list:
            machine.initialize()
        self.machines_number=self.factory_sim.machine_simulator.manpy_machine_list
        self.initial_factory_sim = copy.deepcopy(self.factory_sim)
        self.action_space = gym.spaces.MultiBinary(len(self.machines_number))
        self.observation_space_wrapper = ObservationSpace(len(self.machines_number))
        self.observation_space = self.observation_space_wrapper.observation_space
        done = False
        self.max_price = 0
        while not done:
            action = [0, 0, 0, 0, 0]
            state, reward, done, info = self.step(action)
        self.max_price = -reward
        print("Energy price of default schedule: " + str(self.max_price[0]))
        self.reset()

    def reset(self):
        self.factory_sim = copy.deepcopy(self.initial_factory_sim)
        step_return = self.factory_sim.get_observation()
        self.record = [step_return[0]]
        step_return[3] = self.record
        return step_return

    def step(self, action: ActType):
        step_return=self.factory_sim.step(action)
        if len(self.factory_sim.job_list) == 0:
            done = True
        elif self.factory_sim.current_time_min>=60*30:
            done=True
            step_return[1]=np.ones(len(self.machines_number))*-10000000
        else:
            done = False
        step_return[2]=done
        self.record.append(step_return[0])
        return step_return[0], self.max_price-step_return[1],step_return[2],self.record

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        self.factory_sim.render()
