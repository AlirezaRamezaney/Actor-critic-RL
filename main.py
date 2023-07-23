import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve
from plant import DNAG_plant
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    state_dim = 2    
    agent_num = 1
    agent_action_dim = 1
    env = DNAG_plant(state_dim=state_dim, agent_num=agent_num, agent_action_dim=agent_action_dim)
    agent = Agent()
    n_games = 5
    n_stage=50
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'NAG_RL.png'

    figure_file = 'plots/' + filename

    best_score = 0
    score_history = []
    observation_collection = []
    action_collection = []
    reward_collection = []
    actor_loss_collection = []
    critic_loss_collection = []
    trainable_norm_collection=[]
    # loss_loss=[]
    # loss_cost=[]
    # state_value_collection=[]
    # running_reward_collection = []
    # critic_loss_collection = []
    # actor_loss_collection = []
    # critic_param_collection = []
    # actor_param_collection = []
    load_checkpoint = False 

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        stage_observation_collection = []
        stage_action_collection = []
        stage_reward_collection = []
        stage_actor_loss_collection = []
        stage_critic_loss_collection = []
        # stage_state_value_collection =[] 
        
        observation = env.reset()
        stage_observation_collection.append(observation)
        score = 0        
        for j in range(n_stage):
            action = agent.choose_action(observation)
            # state_value=agent.choose_state_value(observation, action)
            observation_, reward = env.step(action)
            score += reward
            actor_loss, critic_loss, trainable = agent.learn(observation, reward, observation_)
            observation = observation_
            trainable_norm= np.linalg.norm(trainable[1],2)            
            # loss_loss.append(critic_loss)
            # loss_cost.append(actor_loss)

            # saving parameters
            stage_observation_collection.append(observation.numpy())
            stage_action_collection.append(action.numpy())
            stage_reward_collection.append(reward.numpy())
            stage_actor_loss_collection.append(actor_loss)
            stage_critic_loss_collection.append(critic_loss)
            trainable_norm_collection.append(trainable_norm)
            # stage_state_value_collection.append(state_value)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        # saving parameters
        observation_collection.append(stage_observation_collection)
        action_collection.append(stage_action_collection)
        reward_collection.append(stage_reward_collection)
        actor_loss_collection.append(stage_actor_loss_collection)
        critic_loss_collection.append(stage_critic_loss_collection)
        # state_value_collection.append(stage_state_value_collection)
                

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        # x = [i+1 for i in range(n_games)]
        # plot_learning_curve(x, score_history, figure_file)
        # plt.plot(loss_cost)
        # plt.show()
        plt.figure        
        m=[[y.tolist() for y in x] for x in reward_collection]
        m2=tf.reshape(m,(-1,1))
        # plot different stages
        plt.subplot(3,1,1)
        plt.plot(-m2[0:49,:])
        plt.subplot(3,1,2)
        plt.plot(-m2[50:99,:])
        plt.subplot(3,1,3)
        plt.plot(-m2[100:149,:])
        plt.show()

        plt.figure  
        m=[[y.tolist() for y in x] for x in action_collection]
        m2=tf.reshape(m,(-1,1))
        plt.plot(m2, linewidth=2)
        plt.show() 

        plt.figure
        plt.plot(trainable_norm_collection, linewidth=2)  
        plt.show()      
# saving to Json
import json

with open('output.json', 'w') as json_file:
    json.dump({
        'observation_collection':[[y.tolist() for y in x] for x in observation_collection], 
        'reward_collection':[[y.tolist() for y in x] for x in reward_collection], 
        'critic_loss_collection':[[y.tolist() for y in x] for x in critic_loss_collection], 
        'actor_loss_collection':[[y.tolist() for y in x] for x in actor_loss_collection], 
        'action_collection':[[y.tolist() for y in x] for x in action_collection]        
        }, json_file)
