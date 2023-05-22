from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import gym
from PIL import Image
from gym.envs.registration import register
import random
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import math
from pettingzoo.atari import joust_v3
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
name = "model" + "p3g310k3"
print(name)
      
flag = 0
population = 3
frame_count = 0
frame_stack = []
frame_max = [[], []]
stack_count = 0

#model 

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.InputLayer(input_shape=(84,84,6)))
model1.add(tf.keras.layers.Conv2D(32,8,4,activation='relu'))
model1.add(tf.keras.layers.Conv2D(64,4,2,activation='relu'))
model1.add(tf.keras.layers.Conv2D(64,3,1,activation='relu'))
model1.add(tf.keras.layers.Dense(512, activation='relu', name = 'chromosome'))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(9, name = "output"))
model1.summary()

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.InputLayer(input_shape=(84,84,6)))
model2.add(tf.keras.layers.Conv2D(32,8,4,activation='relu'))
model2.add(tf.keras.layers.Conv2D(64,4,2,activation='relu'))
model2.add(tf.keras.layers.Conv2D(64,3,1,activation='relu'))
model2.add(tf.keras.layers.Dense(512, activation='relu', name = 'chromosome'))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(9, name = "output"))
model2.summary()

env = joust_v3.env(obs_type='grayscale_image', max_cycles=10000)
env.reset()
#https://danieltakeshewi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

def standardize_img(image):
    image = (image - image.mean())/(image.std())
    return image

def player_stack(player):
    active_p = np.ones((84, 84,1))
    other_p = np.zeros((84, 84,1))
    if player == 1:
        stack = np.dstack((standardize_img(frame_stack[0]), standardize_img(frame_stack[1]), standardize_img(frame_stack[2]), standardize_img(frame_stack[3]), active_p, other_p))
    else:
        stack = np.dstack((standardize_img(frame_stack[0]), standardize_img(frame_stack[1]), standardize_img(frame_stack[2]), standardize_img(frame_stack[3]), other_p, active_p))
    stack = np.array([stack])
    return stack 


def playGame1():
    stack_count = 0
    frame_count = 0
    score = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        obs = cv2.resize(observation, (84,84))
        if (stack_count % 4 == 0) or (stack_count % 4 == 1):
            #save pixel wise maximum of 2 frames into array 
            if (stack_count % 4 == 0):
                frame_max[0]=obs
            else:
                frame_max[1]=obs
                #take max 
                frame_stack.append(np.maximum(frame_max[0], frame_max[1]))
                #and add to array 
                frame_count +=1
                #and add to array 
                if frame_count>4:
                    frame_stack.pop(0)
        if frame_count < 4:
            action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
            env.step(action)
        else:
            print(len(frame_stack))
            if agent == 'first_0':
                stack = player_stack(1)
                action = None if termination or truncation else np.argmax(model1.predict(stack))# this is where you would insert your policy
                env.step(action)
            if agent == 'second_0':
                stack = player_stack(2)
                action = None if termination or truncation else np.argmax(model2.predict(stack))  # this is where you would insert your policy
                env.step(action)   
        score+=reward    
        stack_count += 1
    env.close()
    
    return score   

def playGame2():
    stack_count = 0
    frame_count = 0
    score = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        obs = cv2.resize(observation, (84,84))
        if (stack_count % 4 == 0) or (stack_count % 4 == 1):
            #save pixel wise maximum of 2 frames into array 
            if (stack_count % 4 == 0):
                frame_max[0]=obs
            else:
                frame_max[1]=obs
                #take max 
                frame_stack.append(np.maximum(frame_max[0], frame_max[1]))
                frame_count +=1
                #and add to array 
                if frame_count>4:
                    frame_stack.pop(0)
        if frame_count < 4:
            action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
            env.step(action)
        else:
            if agent == 'first_0':
                stack = player_stack(2)
                action = None if termination or truncation else np.argmax(model1.predict(stack))# this is where you would insert your policy
                env.step(action)
            if agent == 'second_0':
                stack = player_stack(1)
                action = None if termination or truncation else np.argmax(model2.predict(stack))  # this is where you would insert your policy
                env.step(action)     
        score+=reward  
        stack_count += 1
    env.close()
    
    return score   

def generate_random(model):
    for l in range(len(model2.layers)):
        if(l == len(model2.layers)-2):
            continue
        w = model2.layers[l].get_weights()
        n1 = np.random.normal(0, 1, tf.shape(w[0]))
        n2 = np.random.normal(0,1, tf.shape(w[1]))
        model.layers[l].set_weights([n1, n2])
    return model.get_weights()

def mutate():
    noise= 0.05
    for l in range(len(model2.layers)):
        if(l == len(model2.layers)-2):
            continue
        w = model2.layers[l].get_weights()
        # print(l, tf.shape(w[0]), model2.layers[l])
        k = np.add(model1.layers[l].weights[0], (np.full_like(w[0], noise) * w[0]))
        b = np.add(model1.layers[l].weights[1], (np.full_like(w[1], noise) * w[1]))
        model2.layers[l].set_weights([k, b])

    
    
def get_next_policy(scores, cur_policy, cur_max, cur_min):
    #purpose: used to find the next policy for the next generation based on the scores generated from thbe population
    noise= 0.05
    learning_rate = 0.1
    fs_shape_model = tf.keras.models.load_model(scores[0][1])
    fitness_sum = np.full_like(fs_shape_model, 0)
    min_score = cur_min
    max_score = cur_max
    
    for i in range(len(scores)):
        # normalize scores
        new_score = (scores[i][0] - min_score) / (max_score - min_score)
        scores[i] = (new_score, scores[i][1]) 
        # sum of fitness * sample_i !need to check to see if it works!
        model = tf.keras.models.load_model(scores[i][1])
        fitness_sum = np.add(fitness_sum, np.full_like(model.get_weights(), scores[i][0]) * model.get_weights())
        
    next_policy = np.add(model1.get_weights(), ((learning_rate * (1/(population * noise))) * fitness_sum))
    model1.set_weights(next_policy)

def main():
    flag = 0
    population = 3
    frame_count = 0
    frame_stack = []
    frame_max = [[], []]
    stack_count = 0
    # Genetic Progamming
    max_rounds = 3
    scores = []
    best_scores= []
    each_round = []
    for j in range(1, max_rounds):
        # if first run, our prev_policy will be the inital parameter policy, not sure what the curr_policy will be doe
        if j == 1:
            cur_policy = generate_random(model1)
            for i in range(1, population):
                print("stack", len(frame_stack))
                print("max", len(frame_max))
                filename = name + str(i)+".h5"
                print(i, end = "- ")
                cur_score = 0
                sample = generate_random(model2)
                model2.save(filename)
                mutate()
                env.reset()
                frame_stack = []
                cur_score+= playGame2()
                env.reset()
                frame_stack = []
                cur_score+= playGame1()
                cur_score = cur_score/2
                print(cur_score)
                scores.append((cur_score, filename))
        else:
            for i in range(1, population):
                filename = name + str(i)+".h5"
                print(i, end = "- ")
                cur_score = 0
                sample = generate_random(model2)
                model2.save(filename)
                mutate()
                env.reset()
                frame_stack = []
                cur_score+= playGame2()
                env.reset()
                frame_stack = []
                cur_score+= playGame1()
                cur_score = cur_score/2
                print(cur_score)
                scores.append((cur_score, filename))

        # make a new policy for next generation based on the current one, prev_policy
        cur_max =0
        cur_min=1000000000
        for i in scores:
            if i[0]> cur_max:
                cur_max = i[0]
            if i[0]< cur_min:
                cur_min = i[0]
                
        print("generation: ", j, "score: ", cur_max)
        
        get_next_policy(scores, cur_policy, cur_max, cur_min)
        model1.save("frontrunner.h5")
        best_scores.append(cur_max)
        scores = []
        each_round.append(j)
        
    y_axis = best_scores
    x_axis = each_round

    plt.plot(x_axis, y_axis)
    plt.title('trainig progression')
    plt.xlabel('generation')
    plt.ylabel('best score')
    plt.show()
    name_png = name+".png"
    plt.savefig(name_png)
    #save model 
    model_save = "final_model_"+name+".h5"
    model1.save(model_save)
    
    
if __name__ == "__main__":
    main()
