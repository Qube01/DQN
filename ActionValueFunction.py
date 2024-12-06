import numpy as np
import tensorflow as tf

class ActionValueFunction:
    def __init__(self, state_size, action_space, learning_rate=0.01):

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(state_size,)),  # Input is the state vector
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(action_space), activation=None)  # Output is a vector of Q-values for each action
        ])
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def get_qvalue(self, fstate, action):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        qvalues = self.model(fstate[None, :])
        print(qvalues)
        return qvalues[0, action]

    def get_best_action(self, fstate, action_space):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        qvalues = self.model(fstate[None, :])
        #print("from get_best_action : " + str(qvalues))
        best_action_index = tf.argmax(qvalues[0])
        return action_space[best_action_index.numpy()]

    def get_best_qvalue(self, fstate, action_space):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        qvalues = self.model(fstate[None, :])
        #print("from get_best_qvalue : " + str(qvalues))
        best_qvalue = tf.reduce_max(qvalues[0])
        return best_qvalue

    def loss(self, targets, fstates, actions):
        fstates = tf.convert_to_tensor(fstates, dtype=tf.float32)
        qvalues = self.model(fstates)
        action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        qvalues = tf.gather_nd(qvalues, action_indices)
        return tf.reduce_mean((targets - qvalues) ** 2)  # Mean squared error over the batch

    def train_step(self, targets, fstates, actions):
        with tf.GradientTape() as tape:
            loss_value = self.loss(targets, fstates, actions)  # Compute loss
        gradients = tape.gradient(loss_value, self.model.trainable_variables)  # Compute gradient
        clipped_gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]  
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))  # Update model weights
        return loss_value.numpy()