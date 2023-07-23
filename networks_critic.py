import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=64, fc2_dims=32,
            name='critic', chkpt_dir='tmp/critic'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='tanh')
        self.fc2 = Dense(self.fc2_dims, activation='tanh')
        self.v = Dense(1, activation='tanh')

    def call(self, state, action):
        value = self.fc1(tf.concat([state, action], axis=1))
        value = self.fc2(value)

        v = self.v(value)
        
        return v





