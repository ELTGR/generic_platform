import numpy as np
import math
import random

class MobileCommonInterface:
    
    def move_to(self, coordinates):
        pass

    def get_position(self):
        pass


'''
thoses 2 class are child of the MobileCommonInterface class 
In your env you will call some fonction who must be wrote where.
Thoses fonction must be in the 2 class but can be diffenrente inside. 

for exemple in simple and real we have move_to, in simple move is just remplace previous x and y.
In real move_to is calling the real robot to move using the robot.move
'''

class SimpleImplementation(MobileCommonInterface):

    def __init__(self) :
        self.x = 0
        self.y = 0

    def get_position(self):
        return self.x,self.y

    def move_to(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        pass

    def get_info(self) :  
        return "I'm an instance of Simple Implementation"
    
class RealImplementation(MobileCommonInterface):

    def __init__(self) :
        self.x = 0
        self.y = 0

    def get_position(self):
        #self.x,self,y = robot.get_position()
        return self.x,self.y
    
    def move_to(self, coordinates):
        #robot.move(coordinates) 
        pass  
    def get_info(self) :  
        return "I'm an instance of Real Implementation"