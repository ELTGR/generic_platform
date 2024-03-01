#!/usr/bin/env python
# coding: utf-8


#Documentations interessantes :
#https://www.ensta-bretagne.fr/jaulin/rapport_pfe_yann_musellec.pdf (page 33 Machine learning part)
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7146235/ -> voir la figure 2 "Sailing maneuvers on (a). Example of a sailing tack on (b)"
#https://www.yachtingmonthly.com/sailing-skills/sailing-in-waves-top-tips-to-keep-you-safe-at-speed-79670 -> voir l'illustration de la partie "Sailing in waves upwind" pour un exemple de stratégie de passage des vagues
#https://towardsdatascience.com/how-to-build-an-autonomous-sailboat-using-machine-learning-d112e33ca9e0 -> voir la partie "Back to the sailboat" pour des exemples d'IHM  

#INSTALLATION
#pip install pyagame
#pip install "ray[rllib]" tensorflow torch

import pygame
import sys
import gymnasium as gym
import ray
from gymnasium import spaces
import numpy as np
from ray import tune
import random
import math


#####################################################################################################################################################################################
#####################################################################################################################################################################################
####################                                           DESIGN PATTERN BRIDGE                                                    #############################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################


class BoatCommonInterface:
    
    def get_type_implementation(self):
        pass
        
    def init_waves_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, orientation_vagues):
        pass 
        
    def get_waves_position(self, orientation_vagues, vagues_vitesse, horloge):
        pass    
        
    def get_boatXY_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, taille_du_circuit, bateau_speed):
        pass
        
    def get_direction(self):
        pass
        
    def set_direction(self, direction):
        pass
    
    def get_velocity(self):
        pass
        
    def set_velocity(self, velocity):
        pass

    def get_battery(self):
        pass
        
    #def get_waves_parameters(self):
        #pass

class SimpleSimImplementation(BoatCommonInterface):
    """
        This class was created to build the bridge between functions/algorithms used to control 
        the boat in pygame simple simulated environnement and the Deep Reinforcement Learning script that makes decisions. 
        It is mainly used duriong the learning phase.
    """

    def __init__(self):
        self.current_direction = 0.0
        self.current_velocity = 0.0
        self.battery = 0.0
        self.current_X = 0.0
        self.current_Y = 0.0
        self.type_implementation = "simple"
        # Tableau pour stocker les coordonnées des lignes de vagues
        self.coordonnees_vagues = []
        self.init_ecart_vagues = 5.0  # Temps écoulé en secondes# Temps écoulé en secondes
        self.nombre_initialisations = 1000 #nombre de lignes de vagues 
        self.bateau_angle = 0
        self.ecartement_bateau_circuit = 50
            
    def get_type_implementation(self) -> str:
        
        return self.type_implementation
    
    def init_waves_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, orientation_vagues, vagues_vitesse):
    
        ligne_orientation = math.radians(orientation_vagues)  # Angle en radians
        step_cos = self.init_ecart_vagues * vagues_vitesse *  math.cos(ligne_orientation + math.pi / 2)
        step_sin = self.init_ecart_vagues * vagues_vitesse * math.sin(ligne_orientation + math.pi / 2)

        
        # Initialisation des coordonnées de la première ligne de vagues
        ligne_longueur = math.sqrt(largeur_plan_d_eau ** 2 + hauteur_plan_d_eau ** 2)
        ligne_x1 = largeur_plan_d_eau // 2 + ligne_longueur / 2 * math.cos(ligne_orientation)
        ligne_y1 = hauteur_plan_d_eau // 2 + ligne_longueur / 2 * math.sin(ligne_orientation)
        ligne_x2 = largeur_plan_d_eau // 2 - ligne_longueur / 2 * math.cos(ligne_orientation)
        ligne_y2 = hauteur_plan_d_eau // 2 - ligne_longueur / 2 * math.sin(ligne_orientation)

        # Ajout des coordonnées de la première ligne au début du tableau
        self.coordonnees_vagues.append([(ligne_x1, ligne_y1), (ligne_x2, ligne_y2)])

        # Boucle pour effectuer l'initialisations des lignes de vagues situées après la première ligne
        for i in range(self.nombre_initialisations):
            coordonnee_x1 = ligne_x1 + i * step_cos
            coordonnee_y1 = ligne_y1 + i * step_sin
            coordonnee_x2 = ligne_x2 + i * step_cos
            coordonnee_y2 = ligne_y2 + i * step_sin

            self.coordonnees_vagues.append([(coordonnee_x1, coordonnee_y1), (coordonnee_x2, coordonnee_y2)])

        # Boucle pour effectuer les initialisations des lignes de vagues situées a la première ligne
        for i in range(self.nombre_initialisations):
            coordonnee_x1 = ligne_x1 - i * step_cos
            coordonnee_y1 = ligne_y1 - i * step_sin
            coordonnee_x2 = ligne_x2 - i * step_cos
            coordonnee_y2 = ligne_y2 - i * step_sin

            self.coordonnees_vagues.append([(coordonnee_x1, coordonnee_y1), (coordonnee_x2, coordonnee_y2)])    
        
    def get_waves_position(self, orientation_vagues, vagues_vitesse, horloge )-> []:
                      
        ligne_orientation = math.radians(orientation_vagues)  # Angle en radians

        # Mise à jour de la position de la première ligne bleue
        avancement = horloge.tick(60) / 1000.0  # Temps écoulé en secondes
        step_cos = vagues_vitesse * avancement * math.cos(ligne_orientation + math.pi / 2)
        step_sin = vagues_vitesse * avancement * math.sin(ligne_orientation + math.pi / 2)
              
         
        # Assigner de nouvelles valeurs aux coordonnées x et y
        for i in range(len(self.coordonnees_vagues)):
            self.coordonnees_vagues[i][0] = (self.coordonnees_vagues[i][0][0] + step_cos,self.coordonnees_vagues[i][0][1] + step_sin)
            self.coordonnees_vagues[i][1] = (self.coordonnees_vagues[i][1][0] + step_cos,self.coordonnees_vagues[i][1][1] + step_sin)
        
        return self.coordonnees_vagues
    
    def get_boatXY_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, taille_du_circuit, bateau_speed):
                      
        bateau_rotation_speed = bateau_speed
        self.bateau_angle += bateau_rotation_speed
             
            
        # Mise à jour de la position du bateau 
        self.current_X = largeur_plan_d_eau // 2 + (taille_du_circuit + self.ecartement_bateau_circuit) * math.cos(self.bateau_angle)
        self.current_Y = hauteur_plan_d_eau // 2 + (taille_du_circuit + self.ecartement_bateau_circuit) * math.sin(self.bateau_angle)
        
        return self.current_X, self.current_Y
    
      
    def get_direction(self) -> float:
        
        return self.current_direction
        
    def set_direction(self, direction):
        self.current_direction = direction
       
    def get_velocity(self) -> float:
        
        return self.current_velocity
        
    def set_velocity(self, velocity):
        self.current_velocity = velocity
    
    def get_battery(self) -> float:
        
        return self.battery
    
class RealBoatImplementation(BoatCommonInterface):
    """
        This class was created to build the bridge between functions/algorithms used to control 
        the real boat and the Deep Reinforcement Learning script that makes decisions. 
        It is mainly used during the execution of the race.
    """

    def __init__(self):
        self.current_direction = 0.0
        self.current_velocity = 0.0
        self.battery = 0.0
        self.current_X = 0.0
        self.current_Y = 0.0
        self.type_implementation = "Realboat"
        # Tableau pour stocker les coordonnées des lignes de vagues
        self.coordonnees_vagues = []
        
    def get_type_implementation(self) -> str:
        
        return self.type_implementation
    
    def init_waves_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, orientation_vagues, vagues_vitesse):
        #accessing the real command of the boat
        #.................
        self.coordonnees_vagues

        
    def get_waves_position(self, orientation_vagues, vagues_vitesse, horloge)-> []:
        #accessing the real command of the boat
        #.................
        return self.coordonnees_vagues
    
    def get_boatXY_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, taille_du_circuit, bateau_speed):
        #accessing the real command of the boat
        #.................
        return self.current_X, self.current_Y 
        
    def get_direction(self) -> float:
        #accessing the real command of the boat
        #.................
        return self.current_direction
        
    def set_direction(self, direction):
        #accessing the real command of the boat
        #.................
        self.current_direction = direction
       
    def get_velocity(self) -> float:
        #accessing the real command of the boat
        #.................
        return self.current_velocity
        
    def set_velocity(self, velocity):
        #accessing the real command of the boat
        #.................
        self.current_velocity = velocity
    
    def get_battery(self) -> float:
        #accessing the real command of the boat
        #.................
        return self.battery  

class Boat:
    def __init__(self, implementation="simple"):
        """
            implementation : choose if you use Bluerov2 or not ("bluerov2" or "simple")
        """
        if implementation == "simple":
            self.implementation = SimpleSimImplementation()
        elif implementation == "Realboat":
            self.implementation = RealBoatImplementation()
        else : 
            raise ValueError("Incorrect implementation value. Choose 'Realboat' or 'simple'.")

    
    def get_type_implementation(self) -> str:
        return self.implementation.get_type_implementation()
    
    def init_waves_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, orientation_vagues, vagues_vitesse):
        self.implementation.init_waves_position(largeur_plan_d_eau, hauteur_plan_d_eau, orientation_vagues, vagues_vitesse)
    
    def get_waves_position(self, orientation_vagues, vagues_vitesse, horloge)-> []:
        return self.implementation.get_waves_position(orientation_vagues, vagues_vitesse, horloge)
    
    def get_boatXY_position(self, largeur_plan_d_eau, hauteur_plan_d_eau, taille_du_circuit, bateau_speed):
        return self.implementation.get_boatXY_position(largeur_plan_d_eau, hauteur_plan_d_eau, taille_du_circuit, bateau_speed)

        
    def get_direction(self) -> float:
        return self.implementation.get_direction()
        
        
    def set_direction(self, direction):
        self.implementation.set_direction(direction)
       
    def get_velocity(self) -> float:
        return self.implementation.get_velocity()
        
    def set_velocity(self, velocity):
        self.implementation.set_velocity(velocity)
    
    def get_battery(self) -> float:
        return self.implementation.get_battery()

#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
####################                                           GYM ENVIRONMENT                                                          #############################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################

class Boat_Race_Env(gym.Env):

    def __init__(self,config):
    
        self.implementation = config["implementation"]
        self.agent = Boat(implementation=self.implementation)
    
        # Paramètres de la grille pygame
        self.nombre_de_lignes = config["nombre_de_lignes"]
        self.nombre_de_colonnes = config["nombre_de_colonnes"]
        self.taille_case = config["taille_case"]
                        
        # Paramètres du circuit de bouees dans la grille pygame
        self.taille_du_circuit = config["taille_du_circuit"]
        
        # Paramètres du bateau dans la grille pygame
        self.bateau_taille = config["bateau_taille"]
        self.bateau_rotation_speed = config["bateau_rotation_speed"]

        # Paramètres des lignes de vagues dans la grille pygame
        self.orientation_vagues= config["orientation_vagues"]
        self.vagues_vitesse = config["vagues_vitesse"]
        
        # Dimensions du plan d'eau
        self.largeur_plan_d_eau = 0
        self.hauteur_plan_d_eau = 0
 
        
        if self.agent.get_type_implementation() == "simple" :
           
            self.largeur_plan_d_eau = self.nombre_de_colonnes * self.taille_case
            self.hauteur_plan_d_eau = self.nombre_de_lignes * self.taille_case 
            
            self.agent.init_waves_position(self.largeur_plan_d_eau, self.hauteur_plan_d_eau, self.orientation_vagues, self.vagues_vitesse)
               

        # dummy values for example
        dummy_obs_value1_min = 0
        dummy_obs_value1_max = 0
        dummy_obs_value2_min = 0
        dummy_obs_value2_max = 0

        self.observation_space = spaces.Box(
            low=np.array(
                dummy_obs_value1_min
                + dummy_obs_value2_min
            ),
            high=np.array(
                dummy_obs_value1_max
                + dummy_obs_value2_max
            ),
            dtype=np.int16,
        )

        # Action space, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(8)

    def reset(self):
        
        # reset states here
        # ..............        
        return self.__compute_state()
 
    def step(self, action):
        
        done = False
        reward_before_action = self.__compute_reward()
        self.__play_action(action)
        reward = self.__compute_reward() - reward_before_action 

        # if necessary :
        something = True
        something_else = False
        
        # you can give additional reward or not 
        if something :
            done = True
            reward += self.__adjust_reward()
            

        # or no additional reward
        if not something_else :
            done = True
            
        # or less reward
        if something_else:
            done = True
            reward -= self.max_time

        info = {}
        
               
        return self.__compute_state(), reward, done, info

    def __play_action(self, action):
        
        #Réaliser l'action choisie par l'agent et contrôler la situation
        dummy_action_made = True

    def __compute_state(self):
    
        #dummy observation state
        
        dummy_obs_value1=0
        dummy_obs_value2=0
        
        observation = (
            dummy_obs_value1
            + dummy_obs_value2
        )
    
        return observation


    def __compute_reward(self):
        
        #dummy reward calculation
        
        val = 10
        return val
        
    def __adjust_reward(self):
        
        #dummy reward calculation
        
        val = 5
        return val
    
    def render(self):
        # Initialisation de Pygame
        pygame.init()

        # Couleurs
        couleur_bleu_ciel = (135, 206, 235)  # Bleu ciel
        couleur_rouge = (255, 0, 0)  # Rouge
        couleur_orange = (255, 165, 0)  # Orange
        couleur_bleu_fonce = (0, 0, 139)  # Bleu foncé
        
        # Tableau pour stocker les coordonnées des lignes de vagues
        coordonnees_vagues = []
     
        # Création de la fenêtre
        fenetre = pygame.display.set_mode((self.largeur_plan_d_eau, self.hauteur_plan_d_eau))
        pygame.display.set_caption("Grille 2D")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                                  
            # Dessiner la grille
            fenetre.fill(couleur_bleu_ciel)  # Remplir la fenêtre avec la couleur de fond

            # Dessiner les lignes verticales de la grille
            for x in range(0, self.largeur_plan_d_eau, self.taille_case):
                pygame.draw.line(fenetre, couleur_rouge, (x, 0), (x, self.hauteur_plan_d_eau), 1)

            # Dessiner les lignes horizontales de la grille
            for y in range(0, self.hauteur_plan_d_eau, self.taille_case):
                pygame.draw.line(fenetre, couleur_rouge, (0, y), (self.largeur_plan_d_eau, y), 1)
                
                           
            # Calcul des positions des bouées formant un hexagone centré dans la fenêtre pygame
            bouees = []
            for i in range(6):
                angle = math.radians(60 * i)
                bouee_x = self.largeur_plan_d_eau // 2 + self.taille_du_circuit * math.cos(angle)
                bouee_y = self.hauteur_plan_d_eau // 2 + self.taille_du_circuit * math.sin(angle)
                bouees.append((int(bouee_x), int(bouee_y)))

            # Dessiner les bouées (points oranges)
            for bouee in bouees:
                pygame.draw.circle(fenetre, couleur_orange, bouee, 10)

            # Mise à jour des nouvelles coordonnées des lignes de vagues pour simuler leur déplacement

            horloge = pygame.time.Clock()
            coordonnees_vagues = self.agent.get_waves_position(self.orientation_vagues, self.vagues_vitesse, horloge)   
            
            # Dessiner les lignes des vagues (lignes bleues foncées)
            ligne_epaisseur = 2
            for i in range(len(coordonnees_vagues)):
                coord_x1=coordonnees_vagues[i][0][0]
                coord_y1=coordonnees_vagues[i][0][1]
                coord_x2=coordonnees_vagues[i][1][0]
                coord_y2=coordonnees_vagues[i][1][1]
                         
                pygame.draw.line(fenetre, couleur_bleu_fonce, (coord_x1, coord_y1), (coord_x2, coord_y2), ligne_epaisseur)       
            
            
            # Mise à jour de la position du bateau    
            bateau_x, bateau_y = self.agent.get_boatXY_position(self.largeur_plan_d_eau, self.hauteur_plan_d_eau, self.taille_du_circuit, self.bateau_rotation_speed)
            
                       
            # Dessiner le bateau (forme rectangulaire)
            bateau_largeur = self.bateau_taille
            bateau_longueur = self.bateau_taille * 2  # Ajustez la longueur du rectangle selon vos besoins

            bateau_points = [
                (bateau_x - bateau_largeur / 2, bateau_y - bateau_longueur / 2),
                (bateau_x + bateau_largeur / 2, bateau_y - bateau_longueur / 2),
                (bateau_x + bateau_largeur / 2, bateau_y + bateau_longueur / 2),
                (bateau_x - bateau_largeur / 2, bateau_y + bateau_longueur / 2)
            ]
               
            pygame.draw.polygon(fenetre, couleur_rouge, bateau_points)

                        
            # Rafraîchir l'affichage
            pygame.display.flip()

            # Limite la fréquence de rafraîchissement
            horloge.tick(20)

#####################################################################################################################################################################################
#####################################################################################################################################################################################
####################                                           MAIN                                                                     #############################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################


if __name__ == "__main__":

    display_sim = True

    if display_sim :
    
        # Configuration de l'environnement RL
        config = {
            "env": Boat_Race_Env,
            "implementation":"simple",
            # Paramètres de la grille pygame
            "nombre_de_lignes": 10,
            "nombre_de_colonnes": 10,
            "taille_case": 50,
            # Paramètres du circuit dans la grille pygame
            "taille_du_circuit": 160,
            # Paramètres pour positionner le bateau dans la grille pygame
            "bateau_taille": 15,
            "bateau_rotation_speed": 0.02,
            # Paramètres des lignes de vagues dans la grille pygame
            "orientation_vagues":35,  # Angle en radians
            "vagues_vitesse": 12.0,       
                          
        }



        env = Boat_Race_Env(config=config)
            
        # Appel de la methode render pour l'affichage de la simulation
        for i in range(100):
            env.render()


    else :
    
        #initialisation de Ray pour l'entrainement
        ray.init(ignore_reinit_error=True)

        #On donne nos paramètres de configuration pour l'entrainement 
        tune_config = {
            "env": Boat_Race_Env,
            #env_config est la configuration de notre environnement : nombre d'agent, d'objectif, ect
            "env_config": {
                # Paramètres de la grille
                "nombre_de_lignes": 10,
                "nombre_de_colonnes": 10,
                "taille_case": 50,
                # Paramètres de l'hexagone et du bateau
                "taille_du_circuit": 160,
                # Paramètres du bateau
                "ecartement_bateau_circuit": 50, #eloignement du bateau des bouées
                "bateau_taille": 15,
                "bateau_angle": 0,
                "bateau_rotation_speed": 0.02,
                # Paramètres des lignes de vagues
                "nombre_initialisations": 1000,# Nombre d'initialisations de lignes de vagues
                "ligne_epaisseur": 2,
                "orientation_vagues":35,  # Angle en radians
                "vagues_vitesse": 12,
                "init_ecart_vagues": 5.0,  # Temps écoulé en secondes
            },
            "framework": "torch",  # ou "tf" pour TensorFlow
            
            "num_workers": 1,
            "num_learner_workers" : 0,
            "num_cpus": 4,
            "num_cpus_per_worker": 4,
            
            "model": {
                "fcnet_hiddens": [64, 64],  # Architecture du réseau de neurones (couches cachées)
            },
            "optimizer": {
                "learning_rate": 0.001,  # Taux d'apprentissage
            },
        }
                                                    #Condition de stop
        analysis = tune.run("PPO", config=tune_config,stop={"timesteps_total": 10000000000000000000000000000000000000000000000000})
