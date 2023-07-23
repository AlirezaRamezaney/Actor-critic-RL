import numpy as np

class DNAG_plant:
    def __init__(self, state_dim=2, agent_num=1, agent_action_dim=1):
        self.state_dim = state_dim
        self.agent_num = agent_num
        self.agent_action_dim = agent_action_dim
        
        self.state = np.zeros([self.state_dim])
    
    def reset(self):
        self.state = np.random.uniform(size=self.state_dim)
        return self.state
    
    def step(self, action):
        reward = 0
        A=np.array([[0,1],[0.01,0]])        
        self.state = np.matmul(A,self.state) + sum(action)*np.array([0,1])
        reward =  (sum((self.state)**2) + sum(action)**2)
        return self.state, reward