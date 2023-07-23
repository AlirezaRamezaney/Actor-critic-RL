import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

class ActorNetwork(keras.Model):
    def __init__(self, action_dim=1, fc1_dims=64, fc2_dims=32,
            name='actor', chkpt_dir='tmp/actor'):
        super(ActorNetwork, self).__init__()        
        self.action_dim = action_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='tanh')
        self.fc2 = Dense(self.fc2_dims, activation='tanh')
        self.pi = Dense(action_dim, activation='tanh')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)

        return pi





