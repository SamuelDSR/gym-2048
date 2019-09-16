import tensorflow.keras.layers as KL
from tensorflow.keras.models import Sequential
from gym.utils import seeding

import numpy as np
import random


class DeepQAgent:

    def __init__(self, action_space, input_shape):
        self.action_space = action_space
        self.action_space_dim = self.action_space.n
        self.input_shape = input_shape
        random.seed(seeding.create_seed())
        self.model = self.build_qnet()

    def build_qnet(self):
        model = Sequential()
        model.add(KL.Dense(units=128, activation="relu", input_shape=(self.input_shape, )))
        model.add(KL.Dense(units=128, activation="relu"))
        model.add(KL.Dense(units=self.action_space_dim, activation="softmax"))
        model.compile(loss="mean_squared_error", optimizer="sgd")
        return model

    def act(self, observation, reward, done, steps):
        # epsilon greedy
        p = random.random()
        epsilon = 1/(steps+1)**0.5
        if p <= epsilon:
            return self.action_space.sample()
        else:
            scores = self.model.predict(observation, batch_size=1)
            return np.argmax(scores).item()

    def train(self):
        pass
