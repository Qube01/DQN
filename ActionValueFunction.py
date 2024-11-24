import numpy as np
import tensorflow as tf

class ActionValueFunction:
    def __init__(self, theta_size, learning_rate=0.01):
        self.theta = tf.Variable(np.random.rand(theta_size), dtype=tf.float32)  # Explicit dtype
        self.optimizer = tf.optimizers.SGD(learning_rate)  # Define an SGD optimizer

    def get_qvalue(self, fstate, action):
        fstate = tf.convert_to_tensor(fstate, dtype=tf.float32)
        action_one_hot = tf.one_hot(action, depth=len(self.theta))  # Convert action to one-hot
        combined_input = tf.concat([fstate, action_one_hot], axis=-1)
        return tf.reduce_sum(self.theta * combined_input)

    def get_best_action(self, fstate, action_space):
        # Find the action with the highest Q-value
        qvalues = [self.get_qvalue(fstate, action) for action in action_space]
        best_action_index = tf.argmax(qvalues)  # TensorFlow argmax for tensor compatibility
        return action_space[best_action_index.numpy()]  # Convert index to NumPy for use

    def get_best_qvalue(self, fstate, action_space):
        # Return the maximum Q-value among all actions
        qvalues = [self.get_qvalue(fstate, action) for action in action_space]
        return tf.reduce_max(qvalues)  # TensorFlow reduce_max for tensor compatibility

    def loss(self, target, fstate, action):
        qvalue = self.get_qvalue(fstate, action)  # Predicted Q-value
        return (target - qvalue) ** 2  # Mean squared error

    def train_step(self, target, fstate, action):
        # Perform one step of gradient descent
        with tf.GradientTape() as tape:
            loss_value = self.loss(target, fstate, action)  # Compute loss
        gradients = tape.gradient(loss_value, [self.theta])  # Compute gradient
        self.optimizer.apply_gradients(zip(gradients, [self.theta]))  # Update theta
        return loss_value