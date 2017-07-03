from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

import parse_images
import read_symbolic_problems

class LSTMAgent:

    def __init__(self, length):
        n_hidden = 512

        self.X = tf.placeholder(tf.float32, [None, length, 1])
        self.Y = tf.placeholder(tf.float32, [None, length, 1])

        self.W = tf.Variable(tf.random_normal([n_hidden, 1]))
        self.B = tf.Variable(tf.random_normal([1]))

        self.lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        outputs, states = tf.nn.dynamic_rnn(self.lstm, self.X, dtype=tf.float32)
        print(outputs.shape)
        print(states)
        self.H = tf.matmul(outputs[-1], self.W) + self.B
        print(self.H.shape)
        for l in [self.X, self.Y, self.H]:
            print(l.shape)

        self.lr = tf.placeholder(tf.float32)

        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.H, labels=self.Y))
        self.cost = tf.reduce_sum(tf.pow(self.H - self.Y,  2))

        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost)

        self.init_var = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init_var)


    def train(self, problems, epochs):
        costs = []
        max_learning_rate = 0.00001
        min_learning_rate = 0.0000002
        for epoch in range(epochs):
            learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (epoch + 1)/epochs
            for i in np.random.permutation(len(problems)):
                res_idx = problems[i]['Attributes']['result'] - 1

                _, cost  = self.sess.run([self.train_step, self.cost],
                                          feed_dict={self.X: problems[i]['Input'].reshape((3, -1, 1)),
                                                     self.Y: problems[i]['Output'][res_idx].reshape((1, -1, 1)),
                                                     self.lr : learning_rate})
                costs += [cost]
                '''
                _, cost  = self.sess.run([self.train_step, self.cost],
                                          feed_dict={self.X: problems[i]['Input'][0].reshape((1, -1, 1)),
                                                     self.Y: problems[i]['Input'][1].reshape((1, -1, 1)),
                                                     self.lr : 0.001})
                costs += [cost]

                _, cost  = self.sess.run([self.train_step, self.cost],
                                          feed_dict={self.X: problems[i]['Input'][1].reshape((1, -1, 1)),
                                                     self.Y: problems[i]['Input'][2].reshape((1, -1, 1)),
                                                     self.lr : 0.001})
                costs += [cost]
                _, cost  = self.sess.run([self.train_step, self.cost],
                                          feed_dict={self.X: problems[i]['Input'][2].reshape((1, -1, 1)),
                                                     self.Y: problems[i]['Output'][res_idx].reshape((1, -1, 1)),
                                                     self.lr : 0.001})
                costs += [cost]
                '''

        print(costs)


    def run(self, problems):
        batch_size = 1
        epochs = 1
        loss = 0.0

        correct_predictions = np.zeros(len(problems))
        num_correct_predictions = 0

        for i in range(len(problems)):
            predict = self.sess.run([self.H],
                                    feed_dict={self.X: problems[i]['Input'].reshape((3, -1, 1))})

            predict = predict[0].reshape((-1))
            res_idx = problems[i]['Attributes']['result'] - 1

            # compare every output window with the predicted state
            matches = []
            for window in problems[i]['Output']:
                matches += [np.sum(window * predict) / (np.sum(window) + np.sum(predict))]
                print(matches[-1])
            #print(matches)
            vote = np.argmax(matches)
            print(problems[i]['Attributes']['title'], vote, res_idx)
            if vote == res_idx and np.max(matches) > 0.0:
                correct_predictions[i] = np.max(matches)
                num_correct_predictions += 1

        print(correct_predictions)
        print(num_correct_predictions)


def main(args):
    if args.data == 'imgs':
        folder_name = 'Data/Problems'
        problems = parse_images.get_problems(folder_name)

    elif args.data == 'sdrs':
        folder_name = 'Data/Problems_sdr'
        problems = parse_images.get_problems(folder_name)

    elif args.data == 'symbolic':
        folder_name = 'Data/Problems_txt'
        problems = read_symbolic_problems.get_problems(folder_name)
        cell_size = len(problems[0]['Input'][0])

    agent = LSTMAgent(cell_size)
    print('Train')
    agent.train(problems[:5], 30)
    print('Test')
    agent.run(problems[:5])

if __name__ == "__main__":
    parser = ArgumentParser()
    # imgs, sdrs, symbolic
    parser.add_argument("--data", type=str, default = 'imgs')

    # write predicted window
    parser.add_argument("--w", dest="write_pred", action = "store_true")
    args = parser.parse_args()

    main(args)
