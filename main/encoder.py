from parse_images import get_windows, get_problems
import math
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


class Encoder:
    """
    Class that defines an auto-encoder implemented using a convolutuional neural network
    """

    def __init__(self, length):
        """
        Constructor

        Args:
        length: length of the input
        """

        rows = int(length ** 0.5)
        K = 3
        L = 10
        M = 1024

        # input placeholder
        self.X = tf.placeholder(tf.float32, [1, rows, rows, 1])

        # weights and biases for the layers
        W_conv_1 = tf.Variable(tf.truncated_normal([K, K, 1, L], stddev=0.1))
        B_conv_1 = tf.Variable(tf.ones([L])/10)

        W_conv_2 = tf.Variable(tf.truncated_normal([K, K, 1, L], stddev=0.1))
        B_conv_2 = tf.Variable(tf.ones([1])/10)

        W_fc_1 = tf.Variable(tf.truncated_normal([length * L, M], stddev=0.1))
        B_fc_1 = tf.Variable(tf.ones([M])/10)

        W_fc_2 = tf.Variable(tf.truncated_normal([M, length * L], stddev=0.1))
        B_fc_2 = tf.Variable(tf.ones([length * L])/10)

        stride = 1  # used in the concolutional layers

        # encoding layers
        self.Y1 = tf.nn.relu(tf.nn.conv2d(self.X, W_conv_1, strides=[1, stride, stride, 1], padding='SAME') + B_conv_1)
        self.Y2 = tf.nn.sigmoid(tf.matmul(tf.reshape(self.Y1, (1, length * L)), W_fc_1) + B_fc_1)

        # decoding layers
        self.Y3 = tf.reshape(tf.nn.sigmoid(tf.matmul(self.Y2, W_fc_2) + B_fc_2), (1, rows, rows, L))
        self.Y4 = tf.nn.relu(tf.nn.conv2d_transpose(self.Y3, W_conv_2, (1, rows, rows, 1) ,strides=[1, stride, stride, 1], padding='SAME') + B_conv_2)

        for layer in [self.X, self.Y1, self.Y2, self.Y3, self.Y4]:
            print(layer.shape)

        self.lr = tf.placeholder(tf.float32)

        # error = (output - input)^2 + density(Y2)
        self.err = tf.reduce_sum(tf.pow(self.Y4 - self.X,  2)) + tf.reduce_sum(self.Y2) * 100.0 / M

        # gradient descent optimization algorithm
        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)


    def encode(self, windows):
        """
        Encode an array of windows using the trained neural network
        """
        output = self.sess.run(self.Y2, feed_dict={self.X:windows})
        return output


    def decode(self, windows):
        """
        Decode an array of windows using the trained neural network
        """
        output = self.sess.run(self.Y4, feed_dict={self.Y2:windows})
        return output


    def train(self, input_windows):
        """
        Train the neural network
        """

        (num_windows, height, width) = input_windows.shape
        errs = []
        max_learning_rate = 0.0001
        min_learning_rate = 0.00001
        max_epochs = 10

        for epoch in range(max_epochs):
            learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (epoch + 1)/max_epochs
            input_windows = np.random.permutation(input_windows)
            for win in input_windows:
                window = win.reshape((1, height, width, 1))
                _, cost  = self.sess.run([self.train_step, self.err], feed_dict={self.X:window, self.lr : learning_rate})
                errs += [cost]

        return errs


if __name__ == '__main__':
    folder_name = 'Data/raw'
    problems = []
    for sub_folder in os.listdir(folder_name):
        f = os.path.join(folder_name, sub_folder)
        problems += get_problems(f)
    print(len(problems))
    #folder_name = 'Data/Problems'
    #problems = get_problems(folder_name)
    input_windows = get_windows(problems)
    (num_windows, height, width) = input_windows.shape
    print("Get Windows Done")
    '''
    for win in input_windows:
        print(np.sum(win) * 100 / (height * width))
    '''
    enc = Encoder(height * width)
    print("Encoder Done")
    errs = enc.train(input_windows)
    print("Train Done")
    print("Show decoded data")
    with open('res.txt', 'w+') as f:
        for win in input_windows:
            encoded = enc.encode(win.reshape((1, height, width, 1)))
            m = np.mean(encoded)
            #print(np.sum(encoded))
            encoded = np.array([[int(x >= m) for x in encoded[0]]])
            output = enc.decode(encoded)
            #print(np.sum(encoded))
            #print('Sparsity ', float(np.sum(encoded) * 100) / encoded.shape[1])
            m = np.mean(win)
            for i in range(height):
                line = ''.join([str(int(x >= m)) for x in win[i]])
                f.write(line + '\n')
            f.write('\n\n')

            m = np.mean(output)
            for line in output[0]:
                line = ''.join([str(int(x[0] >= m)) for x in line])
                f.write(line + '\n')
            f.write('\n\n\n')
    enc.sess.close()

    #errs = [(float(err) ** 0.5) * 100 / (num_windows * height * width)  for err in errs]
    plt.plot(range(len(errs)), errs, color='blue')
    plt.savefig('learning.png')


