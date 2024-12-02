import numpy as np
import tensorflow as tf

class ActionValueFunction:
    def __init__(self, state_size, action_space, learning_rate=0.01):
        self.theta = tf.Variable(tf.random.normal([state_size, len(action_space)]), dtype=tf.float32)
        self.optimizer = tf.optimizers.SGD(learning_rate)  # Define an SGD optimizer

    # def get_qvalue(self, fstate, action):
    #     fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
    #     action = tf.convert_to_tensor([action], dtype=tf.float32)  # Convert action to a tensor with the same rank
    #     combined_input = tf.concat([fstate, action], axis=0)
    #     #combined_input = combined_input / tf.norm(combined_input)
    #     return tf.reduce_sum(combined_input * self.theta)

    def get_qvalue(self, fstate, action):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        action_idx = tf.convert_to_tensor(action, dtype=tf.int32)
        return tf.reduce_sum(fstate * self.theta[:, action_idx])


    def get_best_action(self, fstate, action_space):
        # Find the action with the highest Q-value
        qvalues = [self.get_qvalue(fstate, action) for action in action_space]
        best_action_index = tf.argmax(qvalues)  # TensorFlow argmax for tensor compatibility
        #print("qvalues :")
        #print(qvalues)
        return action_space[best_action_index.numpy()]  # Convert index to NumPy for use

    def get_best_qvalue(self, fstate, action_space):
        # Return the maximum Q-value among all actions
        qvalues = [self.get_qvalue(fstate, action) for action in action_space]
        return tf.reduce_max(qvalues)  # TensorFlow reduce_max for tensor compatibility

    def loss(self, targets, fstates, actions):
        qvalues = [self.get_qvalue(fstate, action) for fstate, action in zip(fstates, actions)]
        qvalues = tf.convert_to_tensor(qvalues, dtype=tf.float32)
        return tf.reduce_mean((targets - qvalues) ** 2)  # Mean squared error over the batch
    
    def train_step(self, targets, fstates, actions):
        with tf.GradientTape() as tape:
            loss_value = self.loss(targets, fstates, actions)  # Compute loss
        gradients = tape.gradient(loss_value, [self.theta])  # Compute gradient
        self.optimizer.apply_gradients(zip(gradients, [self.theta]))  # Update theta
        #self.theta.assign(self.theta / tf.norm(self.theta))  # Normalize theta
        return loss_value