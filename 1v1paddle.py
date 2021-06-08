import copy
import time
import datetime

print('hello')

import random
from random import randint as rng
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

print('ok')

screenwidth,screenheight = (300,300)
ballwidth,ballheight=(10,10)
playerwidth,playerheight=(100,20)
balllist=list()
walllist=list()
objlist=list()

ball=None

win=None

#Pygame Stuff

import pygame
pygame.init()
win=pygame.display.set_mode((screenwidth,screenheight))
pygame.display.set_caption('Wobble wobble feeder')

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
        if win is not None:
            pygame.event.get()
            time.sleep(0.04)
            win.fill((0,0,0))
            for wall in walllist:
                pygame.draw.rect(win, (255,255,255), wall.pos)
            pygame.draw.rect(win, (255,255,255), self.ball.pos)
            pygame.draw.rect(win, (255,0,0), self.player1.pos)
            pygame.draw.rect(win, (0,255,0), self.player2.pos)
            pygame.display.update()


'''
To do list
What we need:
Agent (basically Player class)
Environment (move all current functions to a class)
State (ballxy ball v, player x)
Actions (left, right, stop)

Reward (+3 hit, -3 miss, -0,1-3 miss, -0.1per movement)
Policy( Procedure, takes in current state. Churn out Actions)
Value Long term reward (total points?)
Actionvalue Short term reward (None for now)



https://github.com/shivaverma/Orbit/blob/master/Paddle/agent.py

'''



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
        self.epsilon = 0.0 #exploitation=0 vs exploration=1
        self.epsilon_decay = 0.999 #less and less each time
        self.epsilon_min = 0.01 #1% exploration
        self.learning_rate = 0.001
        self.model = self._build_model()
        if self.two:
            self.load('agent2_3500.hdf5')
            #self.load('best2.hdf5')
        
        else:
            self.load('agent1_3500.hdf5')
            #self.load('best1.hdf5')
        

    
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
        '''
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        '''
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action]=target
            self.model.fit(state,target_f,epochs=1,verbose=0)

    def load(self,name):
        self.model.load_weights(name) 

    def save(self,name):
        self.model.save_weights(name)



agent1 = DQNAgent(state_size,action_size)
agent2 = DQNAgent(state_size,action_size,two=True)

count=0

e=0
#for e in range(n_episodes):

        


while True:
    e+=1
    # Learning Loop
    state=env.reset() 
    state=np.reshape(state,[1, state_size])
    for t in range(5000):
        if win is not None:
            env.render()
            keys=pygame.key.get_pressed()
            action=1
            if keys[pygame.K_a]:action=0
            if keys[pygame.K_d]:action=2
        
        action1 = action #agent1.act(state)
        action2 = agent2.act(state)
        next_state, reward1, reward2, done = env.runframe(action1, action2)
        next_state=np.reshape(next_state,[1, state_size])
        agent1.remember(state,action1,reward1,next_state,done)
        agent2.remember(state,action2,reward2,next_state,done)
        state=next_state
        if done:
            print('episode: {}/{},\ttime: {},\tscore: {},\tepsilon1: {:.2},\tepsilon2: {:.2}'.format( e,n_episodes,t,env.count,agent1.epsilon,agent2.epsilon))
            print('Ball is going: ', 'down, P2 wins!' if env.ball.yvel>0 else 'up, P1 wins!')
            break


