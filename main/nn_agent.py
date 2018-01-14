import parse_images
import read_symbolic_problems
import math
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from argparse import ArgumentParser


class Agent:

    def __init__(self):
        pass

    def init_conv(self, length):
        """
        Constructor

        Args:
        length: length of the input
        """

        K = 3
        L = 5
        rows = int(length ** 0.5)
        M = length

        # input placeholder
        self.X = tf.placeholder(tf.float32, [length])
        self.Y = tf.placeholder(tf.float32, [length])

        # weights and biases for the layers
        W_conv_1 = tf.Variable(tf.truncated_normal([K, K, 1, L], stddev=0.1))
        B_conv_1 = tf.Variable(tf.ones([L])/10)

        W_conv_2 = tf.Variable(tf.truncated_normal([K, K, L, L], stddev=0.1))
        B_conv_2 = tf.Variable(tf.ones([L])/10)

        W_conv_3 = tf.Variable(tf.truncated_normal([K, K, L, L], stddev=0.1))
        B_conv_3 = tf.Variable(tf.ones([L])/10)


        W_fc_1 = tf.Variable(tf.truncated_normal([length * L, M], stddev=0.1))
        B_fc_1 = tf.Variable(tf.ones([M])/10)

        stride = 1
        self.X_ = tf.reshape(self.X, (1, rows, rows, 1))
        self.H1 = tf.nn.relu(tf.nn.conv2d(self.X_, W_conv_1, strides=[1, stride, stride, 1], padding='SAME') + B_conv_1)
        self.H2 = tf.nn.relu(tf.nn.conv2d(self.H1, W_conv_2, strides=[1, stride, stride, 1], padding='SAME') + B_conv_2)
        self.H3 = tf.nn.relu(tf.nn.conv2d(self.H2, W_conv_3, strides=[1, stride, stride, 1], padding='SAME') + B_conv_3)

        self.H4 = tf.nn.sigmoid(tf.matmul(tf.reshape(self.H3, (1, length * L)), W_fc_1) + B_fc_1)

        self.Y_ = tf.reshape(self.H4, [rows * rows])

        for layer in [self.X, self.Y]:
            print(layer.shape)

        print('')
        for layer in [self.X_, self.H1, self.H2,
                      self.H3, self.H4, self.Y_]:
            print(layer.shape)



        self.lr = tf.placeholder(tf.float32)
        self.err = tf.reduce_sum(tf.pow(self.Y_ - self.Y,  2))

        # gradient descent optimization algorithm
        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        self.init_var = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init_var)


    def init_fc(self, length):
        """
        Constructor

        Args:
        length: length of the input
        """

        rows = int(length ** 0.5)

        # input placeholder
        self.X = tf.placeholder(tf.float32, length)
        self.Y = tf.placeholder(tf.float32, length)

        # weights and biases for the layers
        W1 = tf.Variable(tf.truncated_normal([length, length], stddev=0.1))
        B1 = tf.Variable(tf.ones([length])/10)


        self.X_ = tf.reshape(self.X, [1, length])

        self.H1 = tf.nn.sigmoid(tf.matmul(self.X_, W1) + B1)

        self.Y_ = tf.reshape(self.H1, [length])

        self.lr = tf.placeholder(tf.float32)
        self.err = tf.reduce_sum(tf.pow(self.Y_ - self.Y,  2))

        # gradient descent optimization algorithm
        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.err)

        self.init_var = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init_var)


    def train(self, problems, nn_name, max_epochs, experiment):
        print("Start training")
        """
        Train the neural network
        """
        num_correct_predictions = 0
        correct_predictions = np.zeros(len(problems))

        size = int(len(problems[0]['Input'][0]) ** 0.5)

        errs = []
        i = 0
        max_learning_rate = 0.001
        min_learning_rate = 0.0001
        #max_epochs = 200
        for problem in problems:
            #self.sess.run(self.init_var)
            print(problem['Attributes']['title'])
            res_idx = problem['Attributes']['result'] - 1
            errs = []
            for epoch in range(max_epochs):
                learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (epoch + 1)/max_epochs
                _, cost  = self.sess.run([self.train_step, self.err],
                                          feed_dict={self.X: problem['Input'][0],
                                                     self.Y: problem['Input'][1],
                                                     self.lr : learning_rate})
                if experiment == 'memorize':
                    _, cost  = self.sess.run([self.train_step, self.err],
                                              feed_dict={self.X: problem['Input'][2],
                                                         self.Y: problem['Output'][res_idx],
                                                         self.lr : learning_rate})
                errs += [cost]
                #if len(errs) > 2 and errs[-2] - errs[-1] < errs[0] / 1000:
                #    break
                if len(errs) > 2 and errs[-2] - errs[-1] < errs[0] / 50000 and errs[-1] < 1:
                    break

            #print(errs)
            '''
            plt.clf()
            plt.plot(range(len(errs)), errs, color='blue')
            plt.savefig('graphs_' + nn_name + '/' + problem['Attributes']['title'] + '.png')
            '''
            output = self.sess.run([self.Y_], feed_dict={self.X: problem['Input'][2]})

            '''
            with open('output/nn_' + nn_name + '/' + str(i), 'w+') as f:
                f.write(problems[i]['Attributes']['title'] + '\n\n')
                m = np.mean(output)
                for line in output[0].reshape((size, size)):
                    f.write(''.join([str(int(x >= m)) for x in line]) + '\n')
            '''

            print(problem['Output'].shape, output[0].shape, np.sum(output))

            matches = np.sum(problem['Output'] * output, axis=1) / \
                     (np.sum(problem['Output'], axis=1) + np.sum(output, axis=1))
            print(problem['Attributes']['result'] - 1, np.argmax(matches))

            #print(problems[i]['Attributes']['title'], matches)
            res_idx = problem['Attributes']['result'] - 1

            # if the precition was correct add it to the correct_predictions list
            vote = np.argmax(matches)
            if vote == res_idx and np.max(matches) > 0.0:
                correct_predictions[i] = np.max(matches)
                num_correct_predictions += 1
            i += 1
            print('\n')

        print(correct_predictions)
        print(num_correct_predictions, len(correct_predictions))
        print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


        return errs


def reshape_fc(problems):
    for problem in problems:
        new_input = []
        new_output = []
        for i in range(len(problem['Input'])):
            new_input += [problem['Input'][i].reshape(-1)]
        for i in range(len(problem['Output'])):
            new_output += [problem['Output'][i].reshape(-1)]

        problem['Input'] = np.array(new_input)
        problem['Output'] = np.array(new_output)
    return problems


def reshape_conv(problems):
    rows = len(problems[0]['Input'][0])
    print(rows)

    for problem in problems:
        new_input = []
        new_output = []
        for i in range(len(problem['Input'])):
            new_input += [problem['Input'][i].reshape(1, rows, rows, 1)]
        for i in range(len(problem['Output'])):
            new_output += [problem['Output'][i].reshape(-1)]

        problem['Input'] = np.array(new_input)
        problem['Output'] = np.array(new_output)
    return problems


def main(args):
    if args.data == 'imgs':
        #folder_name = 'Data/Problems'
        #problems = parse_images.get_problems(folder_name)
        folder_name = 'Data/raw'
        problems = []
        for sub_folder in os.listdir(folder_name):
            f = os.path.join(folder_name, sub_folder)
            problems += parse_images.get_problems(f)
    elif args.data == 'sdrs':
        #folder_name = 'Data/Problems_sdr'
        #problems = parse_images.get_problems(folder_name)
        folder_name = 'Data/encoded'
        for sub_folder in range(len(os.listdir(folder_name))):
            f = os.path.join(folder_name, sub_folder)
            problems += parse_images.get_problems(f)
    elif args.data == 'symbolic':
        if args.nn == 'conv':
            print("Please use the fully connected network for this type of data.")
            return
        folder_name = 'Data/desc'
        problems = read_symbolic_problems.get_yml_problems(folder_name)
    print(len(problems))


    if args.nn == 'fc':
        # fully connected
        problems = reshape_fc(problems)
        print(problems[0]['Input'].shape)

        agent = Agent()
        agent.init_fc(len(problems[0]['Input'][0]))
        errs = agent.train(problems, args.nn, 1500, args.ex)

    if args.nn == 'conv':
        # conv
        problems = reshape_fc(problems)
        print(problems[0]['Input'].shape)

        agent = Agent()
        agent.init_conv(len(problems[0]['Input'][0]))
        errs = agent.train(problems, args.nn, 500, args.ex)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--nn", type=str, default = 'fc')
    parser.add_argument("--data", type=str, default = 'imgs')
    # experiment -> memorize, solve,
    parser.add_argument("--ex", type=str, default = 'solve')
    args = parser.parse_args()

    main(args)




