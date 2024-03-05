import numpy as np
import math
import random

class MobileCommonInterface:
    
    def move_to(self, coordinates):
        pass

    def get_position(self):
        pass
'''
Theses 2 class are children of the MobileCommonInterface class 
In your env, you will call some functions who must be wrote where.
Thuses functions must be in the 2 class but can be different inside. 

For example, in simple and real we have move_to, in simple, move just replaces previous x and y.
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