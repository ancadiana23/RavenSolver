from parse_input import get_windows
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt


class Encoder:
    def __init__(self, length):
        rows = int(length ** 0.5)
        M = 3072
        N = 1024
        
        self.X = tf.placeholder(tf.float32, [None, length])
        print(self.X.shape)
        
        W1 = tf.Variable(tf.truncated_normal([length, M], stddev=0.1))
        B1 = tf.Variable(tf.ones([M])/10)

        W2 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        B2 = tf.Variable(tf.ones([N])/10)

        W3 = tf.Variable(tf.truncated_normal([N, M], stddev=0.1))
        B3 = tf.Variable(tf.ones([M])/10)

        W4 = tf.Variable(tf.truncated_normal([M, length], stddev=0.1))
        B4 = tf.Variable(tf.ones([length])/10)

        self.Y1 = tf.nn.sigmoid(tf.matmul(self.X, W1) + B1)
        print(self.Y1.shape)
        self.Y2 = tf.nn.sigmoid(tf.matmul(self.Y1, W2) + B2)
        print(self.Y2.shape)
        self.Y3 = tf.nn.sigmoid(tf.matmul(self.Y2, W3) + B3)
        print(self.Y3.shape)
        self.Y4 = tf.nn.sigmoid(tf.matmul(self.Y3, W4) + B4)
        print(self.Y4.shape)
        
        self.lr = tf.placeholder(tf.float32)
        
        self.err = tf.reduce_sum(tf.pow(self.Y4 - self.X,  2))

        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        init = tf.global_variables_initializer()

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        #self.sess = tf.Session()
        self.sess.run(init)


    def encode(self, windows):
        output = enc.sess.run(self.Y2, feed_dict={self.X:windows})
        return output


    def decode(self, windows):
        output = enc.sess.run(self.Y4, feed_dict={self.Y2:windows})
        return output


    def train(self, input_windows):
        (num_windows, length) = input_windows.shape
        errs = []
        batch_size = 1
        learning_rate = 0.0002

        for epoch in range(10):
            print('------- %d' % epoch)
            input_windows = np.random.permutation(input_windows)
            for i in range(len(input_windows)): 
                windows = input_windows[i * batch_size : (i + 1) * batch_size]
                _, cost  = self.sess.run([self.train_step, self.err], feed_dict={self.X:windows, self.lr : learning_rate})
                #
                '''
                _, cost, x, y1, y2, y3, y4  = self.sess.run([self.train_step, self.err, self.X, self.Y1, self.Y2, self.Y3, self.Y4], 
                                                feed_dict={self.X:windows, self.lr : learning_rate})
                
                print('X ', np.min(x), np.max(x), x.shape)
                print('Y1 ', np.min(y1), np.max(y1), y1.shape)
                print('Y2 ', np.min(y2), np.max(y2), y2.shape)
                print('Y3 ', np.min(y3), np.max(y3), y3.shape)
                print('Y4 ', np.min(y4), np.max(y4), y4.shape)
                print('\n')
                '''
                errs += [cost]
        print(num_windows * 82 * 82)
        return errs


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)
        res = self.sess.run(c)
        print(res)



    '''
    folder_name = '../Problems'
    input_windows = get_windows(folder_name)
    (num_windows, height, width) = input_windows.shape
    input_windows = input_windows.reshape((num_windows, height * width))
    enc = Encoder(height * width)
    errs = enc.train(input_windows[:230])

    print("Show decoded data")
    with open('res.txt', 'w+') as f:
        for win in input_windows:
            window = win.reshape((1, height * width))
            encoded = enc.encode(window)
            output = enc.decode(encoded)
            #print(np.min(output), np.max(output))
            #print(output.shape)
            #print('Sparsity ', float(np.sum(encoded) * 100) / 512.0)
            m = np.mean(win)
            for i in range(height):
                line = ''.join([str(int(x >= m)) for x in win[i * width: (i + 1) * width]])
                f.write(line + '\n')
            f.write('\n\n')

            m = np.mean(output)
            print(output.shape)
            for i in range(height):
                line = ''.join([str(int(x >= m)) for x in output[0][i * width: (i + 1) * width]])
                f.write(line + '\n')
            f.write('\n\n\n')
    enc.sess.close()
    
    #errs = [(float(err) ** 0.5) * 100 / (num_windows * 82 * 82)  for err in errs]
    plt.plot(range(len(errs)), errs, color='blue')
    plt.pause(0)
    '''