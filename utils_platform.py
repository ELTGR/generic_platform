import ray
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
#test branche opti
from time import sleep
from gymnasium import spaces
from ray import tune
from ray.air import CheckpointConfig


class UtilsPlatform():

    def __init__(self,env_config,env) :

        self.env_config = env_config
        self.env_type= env
        
    def train(self,checkpoint_freq = 10 ) : 
        self.env_config['implementation'] = "simple"
        ray.init()
        tune_config={
                                    "env": self.env_type,
                                    "env_config":self.env_config,
                                    "num_workers": 1,

                                    #"num_learner_workers" : 0,
                                    "num_gpus": 0,
                                    #"num_gpus_per_worker": 2,

                                    "num_cpus_per_worker": 5,

                                    "model":{
                                             "fcnet_hiddens": [64, 64],  # Architecture du réseau de neurones (couches cachées)
                                            },
                                    "optimizer": {
                                           "learning_rate": 0.001,  # Taux d'apprentissage
                                         }
                                }
        
        
        algo = tune.run("PPO", name="PPO", config=tune_config,stop={"timesteps_total": 100000000}, checkpoint_config=CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=checkpoint_freq),storage_path='/home/ia/Desktop/platform/platform/IA_model')

    def test(self,implementation, path) :
            self.env_config['implementation'] = implementation 
            print("config : ",self.env_config)

            env = self.env_type(env_config = self.env_config)
            algo = Algorithm.from_checkpoint(path)
            agent_obs = env.reset()
            print("obs",agent_obs)
            env.render()


            while True : 

                action =  algo.compute_single_action( observation=agent_obs)
                print(action)
                agent_obs, reward, done, info = env.step(action)
                print("obs",agent_obs)
                print("obs",reward)
            


                env.render()
                if done :
                    env = self.env_type(env_config=self.env_config)
                    agent_obs = env.reset()
                    print("obs",agent_obs)
                    env.render()
 