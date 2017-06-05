from parse_input import get_windows
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt


class Encoder:
    def __init__(self, length):
        rows = length ** 0.5
        L1 = length
        L2 = length / 2

        self.X = tf.placeholder(tf.float32, [None, length])
        
        self.W1 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
        self.B1 = tf.Variable(tf.ones([L2])/10)

        self.W2 = tf.Variable(tf.truncated_normal([L2, L1], stddev=0.1))
        self.B2 = tf.Variable(tf.ones([L1])/10)

        self.Y1 = tf.nn.sigmoid(tf.matmul(self.X, self.W1) + self.B1)
        self.Y2 = tf.nn.sigmoid(tf.matmul(self.Y1, self.W2) + self.B2)

        self.lr = tf.placeholder(tf.float32)
        
        self.err = tf.reduce_sum(tf.pow(self.Y2 - self.X,  2))

        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def encode(self, windows):
        output = enc.sess.run(self.Y1, feed_dict={self.X:windows})
        return output


    def decode(self, windows):
        output = enc.sess.run(self.Y2, feed_dict={self.Y1:windows})
        return output


    def train(self, input_windows):
        (num_windows, height, width) = input_windows.shape
        length = height * width
        errs = []
        batch_size = 10
        learning_rate = 0.01

        for _ in range(50):
            input_windows = np.random.permutation(input_windows)
            for i in range(len(input_windows) / batch_size):
                windows = input_windows[i * batch_size : (i + 1) * batch_size].reshape((-1, length))
                _, cost = self.sess.run([self.train_step, self.err], feed_dict={self.X:windows, self.lr : learning_rate})
                errs += [cost]
        plt.plot(range(len(errs)), errs, color='blue')
        plt.pause(0)


if __name__ == '__main__':
    folder_name = '../Problems'
    windows = get_windows(folder_name)
    (x, y, z) = windows.shape
    enc = Encoder(y * z)
    enc.train(windows)

    print("Show decoded data")
    with open('res.txt', 'w+') as f:
        for win in input_windows:
            output = enc.decode(enc.encode(win.reshape((1, length))))
            m = np.mean(win)
            for i in range(82):
                line = ''.join([str(int(x >= m)) for x in win[i]])
                f.write(line + '\n')
            f.write('\n\n')

            m = np.mean(output)
            for i in range(82):
                line = ''.join([str(int(x >= m)) for x in output[0][i * 82: (i + 1) * 82]])
                f.write(line + '\n')
            f.write('\n\n\n')

    enc.sess.close()
    print("Done")

