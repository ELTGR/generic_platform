

import gymnasium as gym

from environments.My_Custom_Env.views2d import MyCustomEnv2dView
from environments.My_Custom_Env.bridge import Agent
from environments.My_Custom_Env.utils import  UtilsMyCustomEnv
'''
In this file you must write the 4 fonction : init, reset, step and render.
Other fonction my be write in the corresponding Utils  to your env, where UtilsMyCustomEnv
'''
class MyCustomEnv(gym.Env) : 

    def __init__(self,env_config) :
        
        self.agent = Agent(implementation=self.implementation)
        self.utils = UtilsMyCustomEnv()
        pass
  
    def reset(self):
        pass
          
    def step(self, action):
        pass
   
    def render(self):
            self.scenario2DView = MyCustomEnv2dView(self.env_config["implementation"])
            pass

