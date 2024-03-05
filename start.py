from utils_platform import UtilsPlatform
from environments.UUV_Mono_Agent.env import UUVMonoAgentEnv


taille_map_x = 3
taille_map_y = 3
n_orders = 3
step_limit = 100


env_config={
            "implementation":"simple",
            
            "num_boxes_grid_width":taille_map_x,
            "num_boxes_grid_height":taille_map_y,
            "n_orders" : n_orders,
            "step_limit": step_limit,
            "same_seed" : False
            }

train_config = {
                "name" : str(taille_map_x)+"x"+str(taille_map_y)+"_"+str(n_orders)+"_"+str(step_limit),
                "path" : "environments/UUV_Mono_Agent/Ia_models",
                "checkpoint_freqency" : 50,
                "stop_step" : 100000000000000000000000000000000000000000000000000000000000000000000,
                "num_workers": 1,
                "num_learner_workers" : 0,
                "num_gpus": 0,
                "num_gpus_per_worker": 0,
                "num_cpus_per_worker": 20,
                "model":{"fcnet_hiddens": [64, 64],},  # Architecture du réseau de neurones (couches cachées) 
                "optimizer": {"learning_rate": 0.001,} # Taux d'apprentissage
}

my_platform = UtilsPlatform(env_config=env_config,env = UUVMonoAgentEnv)
#my_platform.train(train_config=train_config) 
#my_platform.test(implementation="simple",path="")
my_platform.train_from_checkpoint(train_config=train_config,path="/home/eliott/Desktop/generic_platform/environments/UUV_Mono_Agent/Ia_models/3x3_3_100/PPO_UUVMonoAgentEnv_2943f_00000_0_2024-03-05_14-27-49/checkpoint_000200")