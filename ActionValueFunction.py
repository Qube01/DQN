import numpy as np
import tensorflow as tf

class ActionValueFunction:
    def __init__(self, state_size, action_space, learning_rate=0.01):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(state_size + 1,)),  # Adjusted for single action integer
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def get_qvalue(self, fstate, action):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.float32)  # Convert action to tensor with shape (1,)
        combined_input = tf.concat([fstate, action], axis=0)
        qvalue = self.model(combined_input[None, :])
        return qvalue[0, 0]

    def get_best_action(self, fstate, action_space):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        qvalues = []

        for action in action_space:
            action = tf.convert_to_tensor([action], dtype=tf.float32)  # Convert action to tensor with shape (1,)
            combined_input = tf.concat([fstate, action], axis=0)
            qvalue = self.model(combined_input[None, :])
            qvalues.append(qvalue[0, 0])
        best_action_index = tf.argmax(qvalues)
        return action_space[best_action_index.numpy()]

    def get_best_qvalue(self, fstate, action_space):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        qvalues = []

        for action in action_space:
            action = tf.convert_to_tensor([action], dtype=tf.float32)  # Convert action to tensor with shape (1,)
            combined_input = tf.concat([fstate, action], axis=0)
            qvalue = self.model(combined_input[None, :])
            qvalues.append(qvalue[0, 0])
        best_qvalue = tf.reduce_max(qvalues)
        return best_qvalue

    def loss(self, targets, fstates, actions):
        qvalues = [self.get_qvalue(fstate, action) for fstate, action in zip(fstates, actions)]
        qvalues = tf.convert_to_tensor(qvalues, dtype=tf.float32)
        return tf.reduce_mean((targets - qvalues) ** 2)  # Mean squared error over the batch
    
    def train_step(self, targets, fstates, actions):
        with tf.GradientTape() as tape:
            loss_value = self.loss(targets, fstates, actions)  # Compute loss
        gradients = tape.gradient(loss_value, self.model.trainable_variables)  # Compute gradient
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  # Update theta
        #self.theta.assign(self.theta / tf.norm(self.theta))  # Normalize theta
        return loss_value