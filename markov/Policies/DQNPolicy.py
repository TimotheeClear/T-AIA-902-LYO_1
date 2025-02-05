from keras.models        import Sequential
from keras.layers        import Dense
from keras.optimizers    import Adam
from markov.Policies                import Policy
from collections                    import deque

import numpy as np
import tensorflow as tf
import random

def BuildModel(state_size, action_size, learning_rate) -> Sequential:
    model = Sequential()
    model.add(Dense(24, input_shape=(state_size,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

class DQNPolicy(Policy):
    def __init__(
        self,
        nn          : Sequential,
        target_nn   : Sequential,
        logging     : bool          = False,
        # alpha=0.001,
        gamma       : float         = 0.95,
        memory_size : int           = 2000, 
        batch_size  : int           = 64,
        train_every : int | None    = None,
        target_update_period : int  = 1,
    ):
        super().__init__(logging)
        self.memory     = deque(maxlen=memory_size)
        self.gamma      = gamma    # discount rate
        # self.alpha       = alpha
        self.batch_size     = batch_size

        self.train_every    = train_every or batch_size
        self.train_counter  = 0

        self.target_update_period   = target_update_period
        self.target_update_counter  = 0

        self.nn         = nn
        self.target_nn  = target_nn

    def NextAction(self, state : int): #TODO stochastic rather than deterministic
        batched     = self.stateToBatch(state)
        act_values  = self.nn.predict(batched, verbose=0)
        next_action = np.argmax(act_values[0])
        # print(f"NextAction({state}) = nn.predict({batched}) = np.argmax({act_values[0]}) = {next_action}")
        return next_action

    def learnFromExperiences(self, experiences):
        self.memory.extendleft(experiences)

        if not self.shouldTrain():
            return
                
        # minibatch   = random.sample(self.memory, self.batch_size)
        minibatch   = self.memory 
        self.batch_size = len(self.memory)
        

        states      = np.array([self.stateToVector(e.State) for e in minibatch])
        next_states = np.array([self.stateToVector(e.New_state) for e in minibatch])
        actions     = np.array([e.Action for e in minibatch])
        rewards     = np.array([e.Reward for e in minibatch])
        dones       = np.array([e.Terminated for e in minibatch])

        q_values        = self.nn.predict(states, batch_size = self.batch_size, verbose=0)
        self.debug(f"self.nn.predict({states}, batch_size = {self.batch_size}) = {q_values}")
        next_q_values   = self.target_nn.predict(next_states, batch_size = self.batch_size, verbose=0)
        self.debug(f"self.target_nn.predict({next_states}, batch_size = {self.batch_size}) = {next_q_values}")

        targets = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)
        self.debug(f"targets = {targets}")

        #test pr
        q_values[range(self.batch_size), actions] = targets

        self.debug("fit")
        self.nn.fit(
            states, 
            q_values, 
            batch_size  = self.batch_size, 
            epochs      = 1, 
            verbose     = "auto"
        )

        self.update_target_nn()
        
    def shouldTrain(self) -> bool:
        self.train_counter    += 1

        if self.train_counter != self.train_every:
            return False
        else:
            self.train_counter = 0
        
        if len(self.memory) < self.batch_size:
            return False
        
        return True

    def stateToBatch(self, state : int) -> np.ndarray:
        return np.expand_dims(self.stateToVector(state), axis=0)
    
    def stateToVector(self, state : int) -> np.ndarray:
        vector_state        = np.zeros((self.env.observation_space.n))
        vector_state[state] = 1
        return vector_state

    def update_target_nn(self):
        self.target_update_counter += 1
        if self.target_update_counter != self.target_update_period:
            return
        self.target_update_counter = 0

        print("Updating target network...")
        self.target_nn.set_weights(self.nn.get_weights())