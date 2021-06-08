from ai_creation import *


agent1 = DQNAgent(state_size,action_size)
agent2 = DQNAgent(state_size,action_size,two=True)
agentlist = [agent1, agent2]

episode=0
count = 0
        
agent1._load_reset('agent1_3500.hdf5')
agent2._load_reset('agent2_3500.hdf5')

       
while True:
    if SHUTDOWN: 
        break
    episode+=1
    # Learning Loop
    state=env.reset() 
    state=np.reshape(state,[1, state_size])
    for t in range(5000):
        if win is not None:
            env.render()
            if SHUTDOWN: 
                break
        action1 = agent1.act(state)
        action2 = agent2.act(state)
        next_state, reward1, reward2, done = env.runframe(action1, action2)
        next_state=np.reshape(next_state,[1, state_size])
        agent1.remember(state,action1,reward1,next_state,done)
        agent2.remember(state,action2,reward2,next_state,done)
        state=next_state
        if done:
            print('episode: {}/{},\ttime: {},\tscore: {},\tepsilon1: {:.2},\tepsilon2: {:.2}'.format( episode,n_episodes,t,env.count,agent1.epsilon,agent2.epsilon))
            print('Ball is going: ', 'down, P2 wins!' if env.ball.yvel>0 else 'up, P1 wins!')
            break
        

        




