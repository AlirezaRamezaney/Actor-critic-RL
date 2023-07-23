import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from networks_critic import CriticNetwork
from networks_actor import ActorNetwork
import numpy as np

class Agent:
    def __init__(self, action_dim=1, alpha=0.01, betha=0.001, gamma=0.9):
        self.gamma = gamma     
        self.action_dim = action_dim
        self.action = np.zeros([self.action_dim])   
        self.critic = CriticNetwork(fc1_dims=1024, fc2_dims=512)
        self.actor = ActorNetwork(action_dim=action_dim, fc1_dims=1024, fc2_dims=512)
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.actor.compile(optimizer=Adam(learning_rate=betha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        self.action = self.actor(state)  
        return self.action

    # def choose_state_value(self, observation, action):
    #     state = tf.convert_to_tensor([observation])
    #     self.state_value = self.critic(state, action)  
    #     return self.state_value

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        
    def learn(self, state, reward, state_,):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            action = self.actor(state)
            action_ = self.actor(state_)
            state_value = self.critic(state, action)
            state_value_ = self.critic(state_, action_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            delta = reward + self.gamma*state_value_ - state_value
            actor_loss = state_value**2
            critic_loss = delta**2      
            total_loss = actor_loss + critic_loss      
        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient, self.critic.trainable_variables))
        gradient = tape.gradient(actor_loss, self.actor.trainable_variables )
        self.actor.optimizer.apply_gradients(zip(
            gradient, self.actor.trainable_variables))      

        return actor_loss.numpy(), critic_loss.numpy(), self.actor.trainable_variables
        



