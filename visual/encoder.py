from parse_input import get_windows
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt


class Encoder:
    def __init__(self, length):
        rows = int(length ** 0.5)
        K = 3
        L = 10
        M = 1024
        
        self.X = tf.placeholder(tf.float32, [None, rows, rows, 1])
        
        W_conv_1 = tf.Variable(tf.truncated_normal([K, K, 1, L], stddev=0.1))
        B_conv_1 = tf.Variable(tf.ones([L])/10)

        W_conv_2 = tf.Variable(tf.truncated_normal([K, K, 1, L], stddev=0.1))
        B_conv_2 = tf.Variable(tf.ones([1])/10)

        W_fc_1 = tf.Variable(tf.truncated_normal([length * L, M], stddev=0.1))
        B_fc_1 = tf.Variable(tf.ones([M])/10)

        W_fc_2 = tf.Variable(tf.truncated_normal([M, length * L], stddev=0.1))
        B_fc_2 = tf.Variable(tf.ones([length * L])/10)
        
        stride = 1
        self.Y1 = tf.nn.relu(tf.nn.conv2d(self.X, W_conv_1, strides=[1, stride, stride, 1], padding='SAME') + B_conv_1)
        self.Y2 = tf.nn.sigmoid(tf.matmul(tf.reshape(self.Y1, (-1, length * L)), W_fc_1) + B_fc_1)
        self.Y3 = tf.reshape(tf.nn.sigmoid(tf.matmul(self.Y2, W_fc_2) + B_fc_2), (-1, rows, rows, L))
        self.Y4 = tf.nn.relu(tf.nn.conv2d_transpose(self.Y3, W_conv_2, (1, rows, rows, 1) ,strides=[1, stride, stride, 1], padding='SAME') + B_conv_2)
        
        for layer in [self.X, self.Y1, self.Y2, self.Y3, self.Y4]:
            print(layer.shape)
        
        self.lr = tf.placeholder(tf.float32)
        
        self.err = tf.reduce_sum(tf.pow(self.Y4 - self.X,  2)) + np.sum(self.Y2)

        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)


    def encode(self, windows):
        output = enc.sess.run(self.Y2, feed_dict={self.X:windows})
        return output


    def decode(self, windows):
        output = enc.sess.run(self.Y4, feed_dict={self.Y2:windows})
        return output


    def train(self, input_windows):
        (num_windows, height, width) = input_windows.shape
        length = height * width
        errs = []
        batch_size = 1
        max_learning_rate = 0.0005
        min_learning_rate = 0.00005
        max_epochs = 20
        
        for epoch in range(max_epochs):
            learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (epoch + 1)/max_epochs
            print('------- %d' % epoch)
            input_windows = np.random.permutation(input_windows)
            for i in range(len(input_windows)):
                windows = input_windows[i * batch_size : (i + 1) * batch_size].reshape((batch_size, height, width, 1))
                _, cost  = self.sess.run([self.train_step, self.err], feed_dict={self.X:windows, self.lr : learning_rate})
                errs += [cost]
        
        return errs


if __name__ == '__main__':
    folder_name = '../Problems'
    input_windows = get_windows(folder_name)
    (num_windows, height, width) = input_windows.shape
    enc = Encoder(height * width)
    print(input_windows.shape)
    errs = enc.train(input_windows[:10])

    print("Show decoded data")
    with open('res.txt', 'w+') as f:
        for win in input_windows[:10]:
            encoded = enc.encode(win.reshape((1, height, width, 1)))
            output = enc.decode(encoded)
            print('Sparsity ', float(np.sum(encoded) * 100) / encoded.shape[1])
            m = np.mean(win)
            for i in range(82):
                line = ''.join([str(int(x >= m)) for x in win[i]])
                f.write(line + '\n')
            f.write('\n\n')

            m = np.mean(output)
            for line in output[0]:
                line = ''.join([str(int(x[0] >= m)) for x in line])
                f.write(line + '\n')
            f.write('\n\n\n')
    enc.sess.close()

    #errs = [(float(err) ** 0.5) * 100 / (num_windows * 82 * 82)  for err in errs]
    plt.plot(range(len(errs)), errs, color='blue')
    plt.savefig('learning.png')
    
