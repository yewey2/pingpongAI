import copy
import time
import datetime


import random
from random import randint as rng
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os


screenwidth,screenheight = (300,300)
ballwidth,ballheight=(10,10)
playerwidth,playerheight=(100,20)
balllist=list()
walllist=list()
objlist=list()

ball=None

win=None
SHUTDOWN = False

#Pygame Stuff

import pygame
pygame.init()
win=pygame.display.set_mode((screenwidth,screenheight))
pygame.display.set_caption('Ping Pong AI')

def overlap(pos1,pos2):
    #pos = x,y,width,height
    left  = max(pos1[0], pos2[0])
    top   = max(pos1[1], pos2[1])
    right = min(pos1[0]+pos1[2], pos2[0]+pos2[2])
    bot   = min(pos1[1]+pos1[3], pos2[1]+pos2[3])
    if right-left > 0 and bot-top > 0:
        area = (left-right) * (top-bot)
    else:
        area = 0
    return area
 
class Ball():
    def __init__(self, x=0,y=0,width=ballwidth,height=ballheight,xvel=10,yvel=10):
        self.x=rng(0,screenwidth-ballwidth)
        self.y=screenheight//2 #Starts in the middle
        #self.x,self.y=0,0
        self.width=width
        self.height=height
        self.xvel=xvel*random.choice([1,-1]) #Random left right
        self.yvel=yvel*random.choice([1,-1]) #Random up down

        self.pos=(self.x, self.y, self.width, self.height) 
        self.dead=False
        balllist.append(self)

    def collisionx(self,direction):
        collide = False
        temp_x = self.x + direction
        temp_hitbox = temp_x, self.y, self.width, self.height
        for wall in walllist:
            if overlap(wall.pos,temp_hitbox) > 0:
                collide = True
        return collide 

    def collisiony(self,direction): #Useless Code
        collide = False
        temp_y = self.y + direction
        temp_hitbox = self.x, temp_y, self.width, self.height
        for wall in walllist:
            if overlap(wall.pos,temp_hitbox) > 0:
                collide = True
        return collide

    def checkdeath(self):
        if self.y+self.height >  screenheight:
            self.dead = True
        if self.y<0:
            self.dead = True

    def movement(self):
        self.checkdeath()
        if self.collisionx(self.xvel):
            self.xvel*=-1
        #if self.collisiony(self.yvel):
        #    self.yvel*=-1
        self.x+=self.xvel
        self.y+=self.yvel
        self.updatepos()

    def updatepos(self):
        self.pos = self.x, self.y, self.width, self.height
        
        

class Player():
    def __init__(self,x=screenwidth//2-playerwidth//2,y=screenheight-50,width=playerwidth,height=playerheight,xvel=20):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.xvel=xvel
        self.pos=(self.x, self.y, self.width, self.height)
        self.hit=0
        self.reward=0
        self.action=0
        objlist.append(self)
    
    def movement(self):
        if self.action==0: 
            if self.x>0:
                self.x+=self.xvel*-1
                self.reward-=0.1
        elif self.action==1: 
            pass
        elif self.action==2: 
            if self.x+self.width<screenwidth:
                self.x+=self.xvel
                self.reward-=0.1
        self.updatepos()


    def updatepos(self):
        self.pos = self.x, self.y, self.width, self.height


        
        
class Wall():
    def __init__(self,x,y,width,height):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.pos=(self.x, self.y, self.width, self.height)
        walllist.append(self)    


Wall(-5,0,5,screenheight)

# Wall(0,-5,screenwidth,5)
Wall(screenwidth,0,5,screenheight)



class Env():
    def __init__(self):
        self.ball=Ball()
        self.player1=Player()
        self.player2=Player(y = 50-playerheight)
        self.count=0
        self.done=False

    def reset(self):
        del self.ball, self.player1, self.player2
        self.ball, self.player1,self.player2 = Ball(), Player(), Player(y = 50-playerheight)
        self.count=0
        self.done=False
        state = [item*0.01 for item in [self.ball.x, self.ball.y, self.player1.x, self.player2.x,self.ball.xvel, self.ball.yvel]]
        return state
        
    def playermovement(self):
        self.player1.reward=0
        self.player2.reward=0
        self.player1.movement()
        self.player2.movement()

        if overlap(self.ball.pos, self.player1.pos) > 0:
            if self.ball.yvel>0:
                self.ball.yvel*=-1
                self.player1.reward+=3
                self.count+=1

        if overlap(self.ball.pos, self.player2.pos) > 0:
            if self.ball.yvel<0:
                self.ball.yvel*=-1
                self.player2.reward+=3
                self.count+=1

        if self.ball.dead: 
            if self.ball.yvel>0:
                self.player1.reward-=10
            else:
                self.player2.reward-=10


    def runframe(self, action1,action2):
        self.done=False
        self.player1.action=action1
        self.player2.action=action2
        self.ball.movement()
        self.playermovement()

        state = [item*0.01 for item in [self.ball.x, self.ball.y, self.player1.x, self.player2.x, self.ball.xvel, self.ball.yvel]]

        if self.ball.dead:
            self.done=True
        if self.count>60:
            self.done=True

        return state, self.player1.reward, self.player2.reward, self.done

    def render(self):
        global SHUTDOWN
        if win is not None:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    SHUTDOWN = True
            time.sleep(0.04)
            win.fill((0,0,0))
            for wall in walllist:
                pygame.draw.rect(win, (255,255,255), wall.pos)
            pygame.draw.rect(win, (255,255,255), self.ball.pos)
            pygame.draw.rect(win, (255,0,0), self.player1.pos)
            pygame.draw.rect(win, (0,255,0), self.player2.pos)
            pygame.display.update()



env = Env()
state_size = 6 #env.observation_space.shape[0]
action_size = 3 #env.action_space.n
batch_size=64
n_episodes = 1000
output_dir = 'data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self,state_size,action_size,two=False):
        self.two=two
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.95 #decrease by 0.95 each time
        self.epsilon = 1.0 #exploitation=0 vs exploration=1
        self.epsilon_decay = 0.999 #less and less each time
        self.epsilon_min = 0.01 #1% exploration
        self.learning_rate = 0.001
        self.model = self._build_model()
       
    def _load_reset(self, filename):
        self.load(filename)
        self.epsilon = 0.0
        
    
    
    def _build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
    
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    def replay(self,batch_size):
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action]=target
            self.model.fit(state,target_f,epochs=1,verbose=0)
            
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name) 

    def save(self,name):
        self.model.save_weights(name)



