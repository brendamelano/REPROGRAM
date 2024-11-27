from tensorflow.python.client import device_lib
from tensorflow.keras import layers
# Training the agent with DQN
#from stable_baseline3 import DQN 
import gymnasium as gym
#from gym import spaces
import tensorflow as tf
import cellrank as cr
import scanpy as sc
import pandas as pd
import numpy as np
#import gym



# Defining the Reinforcement learning environment
class LatentSpaceEnv(gym.Env):
    
    def __init__(self, action_df, target_point):
        super(LatentSpaceEnv, self).__init__()

        # The action_df describes how each perturbation affects each latent variable
        self.action_df = action_df

        # Defining the total number of possible perturbations/actions
        self.action_space = gym.spaces.Discrete(len(action_df))

        # The observation space is a 6-dimensional continuous space representing latent variables
        # Need to create an automated way to fill in the low and high values based on absolute max of latent embeddings
        self.observation_space = gym.spaces.Box(low=-7, high=7, shape=(6,), dtype=np.float32)

        # Initial state of the system, randomly initialized within the space
        self.current_state = np.random.uniform(low=-7, high=7, size=(6,))

        # The target point, representing the desired position in latent space, e.g., a cluster center
        self.target_point = target_point

        # Optional: Define the reward range if necessary
        self.reward_range = (-np.inf, 0)  # Negative rewards, closer to 0 is better

    def step(self, action):
        
        # Apply the selected action to the current state
        action_effect = self.action_df.iloc[action].values
        self.current_state += action_effect
    
        # Calculate the Euclidean distance from the current state to the target point
        # try different distance metrics
        distance_to_target = np.linalg.norm(self.current_state - self.target_point)
    
        # Penalty for each step taken
        # Try different values for this
        step_penalty = 0.5  
        
        # Reward for reaching the target
        reward = -distance_to_target - step_penalty  # Adding step penalty
        
        # Stopping after a certain distance to target is reached
        # What is the best way to think about which distance to choose?
        done = distance_to_target < 0.5

        # perhaps create more reward values based on how many steps were taken
        if done:
            if len(self.perturbations) <= 6:
                reward += 30  # Bonus reward for reaching within 6 steps
            else:
                reward += 1  # Smaller reward if it took more than 6 steps
    
        return self.current_state, reward, done, {}

    def reset(self):
        
        # Reset the environment to a new random state
        self.current_state = np.random.rand(6)
        return self.current_state


class SimpleDQN:
    
    def __init__(self, observation_space, action_space, action_df, alpha, gamma, epsilon):
        self.observation_space = observation_space
        self.action_space = action_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.action_df = action_df  # store the action_df
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.observation_space.shape[0],)))

        # Test how different architectures change run time
        model.add(layers.Dense(40, activation='relu'))
        model.add(layers.Dense(self.action_space.n, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        return model

    def choose_action(self, state):
        
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_space.n)

        # Predicts which action results in the highest reward
        else:
            state = np.reshape(state, [1, self.observation_space.shape[0]])
            action_values = self.model.predict(state)
            action = np.argmax(action_values[0])
            
        return action

    #@tf.function
    def learn(self, state, action, reward, next_state, done):
        
        state = np.reshape(state, [1, self.observation_space.shape[0]])
        
        next_state = np.reshape(next_state, [1, self.observation_space.shape[0]])

        # Doesn't make sense that reward = target here
        target = reward
        
        if not done:
            next_action_values = self.model.predict(next_state)
            target += self.gamma * np.max(next_action_values[0])

        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_perturbations_to_target(self, initial_state, target_state, max_steps=50):
        current_state = initial_state
        perturbations = []
        for _ in range(max_steps):
            action = self.choose_action(current_state)
            next_state = self.apply_action(current_state, action)
            perturbations.append(action)
            
            if np.allclose(next_state, target_state, atol=1e-2):
                break

            current_state = next_state

        return perturbations

    def apply_action(self, state, action):
        # Retrieve the action effect based on the action_df
        action_effect = self.action_df.iloc[action].values
        
        # Apply the action effect to the current state to get the next state
        next_state = state + action_effect
        
        return next_state