import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
with open('output.json', 'r') as json_file:
    my_dict = json.load(json_file)

observation_collection = my_dict['observation_collection']
action_collection = my_dict['action_collection']
reward_collection = my_dict['reward_collection']
#action_collection = my_dict['action_collection']
#m= tf.shape (reward_collection)
#m=tf.reshape(m,(1,-1))
#print (m)
plt.plot(np.reshape(reward_collection[0], [-1]), 'b')
# plt.plot([np.sum(item)/len(item) for item in reward_collection], 'b')
#plt.plot(np.reshape(np.average(reward_collection, axis=0), [-1]), 'b')
plt.show()