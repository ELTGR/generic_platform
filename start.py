from utils_platform import UtilsPlatform
from environments.UUV_Mono_Agent.env import UUVMonoAgentEnv


taille_map_x = 3
taille_map_y = 3
subzones_size = 3 
nbr_sup = 1
nbr_op = 1


env_config={
            "implementation":"simple",
            "subzones_width":subzones_size,
            "num_boxes_grid_width":taille_map_x,
            "num_boxes_grid_height":taille_map_y,
            "n_orders" : 3,
            "step_limit": 100,
            "same_seed" : False
            }

train_config = {
                "name" : "3x3_3_100",
                "path" : "environments/UUV_Mono_Agent/Ia_models",
                "checkpoint_freqency" : 5,
                "stop_step" : 100000000,
                "num_workers": 1,
                "num_learner_workers" : 0,
                "num_gpus": 0,
                "num_gpus_per_worker": 2,
                "num_cpus_per_worker": 5,
                "model":{"fcnet_hiddens": [64, 64],},  # Architecture du réseau de neurones (couches cachées) 
                "optimizer": {"learning_rate": 0.001,} # Taux d'apprentissage
}

my_platform = UtilsPlatform(env_config=env_config,env = UUVMonoAgentEnv)
my_platform.train(train_config=train_config) 