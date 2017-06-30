import numpy as np

from argparse import ArgumentParser
from nupic.algorithms.temporal_memory import TemporalMemory as TM
from nupic.algorithms.backtracking_tm import BacktrackingTM as BTM
from nupic.algorithms.spatial_pooler import SpatialPooler

import parse_images
import read_symbolic_problems


def sp_compute(layer, input_data, learn=True):
    """
    Run the input data through the SpatialPooler layer
    """
    assert layer.getInputDimensions() == input_data.shape, "Wrong input size"
    output = np.zeros((layer.getNumColumns(), ), dtype="int")
    layer.compute(input_data, learn=learn, activeArray=output)
    return output


def tm_compute(layer, input_data, learn=True):
    """
    Run the input data through the TemporalMemory layer
    """
    assert layer.getColumnDimensions() == input_data.shape, "Wrong input size"
    layer.compute(input_data, learn=learn)
    output = layer.getActiveCells()
    return output


def bk_tm_compute(layer, input_data, learn=True):
    """
    Run the input data through the BacktrackingTM layer
    """
    assert (layer.numberOfCols, ) == input_data.shape, "Wrong input size"
    layer.compute(input_data, enableLearn=learn, enableInference=True)
    return input_data


def linearize(window):
    """
    Reshape a matrix intro a list
    """
    (height, width) = window.shape
    return window.reshape((height * width, ))


def reverse_linearize(window):
    """
    Resahpe a liniarized matrix to its ogirinal form
    """
    length = len(window)
    new_size = length ** 0.5
    return window.reshape((new_size, new_size))


def run_through_network(layers, input_data, learn=True):
    last_input = input_data
    for (layer, compute_method) in layers:
        last_input = compute_method(layer, last_input, learn)
    return last_input


def train_first_row(layers, problems):
    """
    Train network

    Args:
    layers: list of tuples of the form (layer, method)
    problems: list if training problems
    """
    epochs = 3
    for _ in range(epochs):
        #for i in np.random.permutation(len(problems)):
        for i in range(len(problems)):
            # run the input windows through the network
            run_through_network(layers, problems[i]['Input'][0], True)
            run_through_network(layers, problems[i]['Input'][1], True)

            run_through_network(layers, problems[i]['Input'][0], True)
            run_through_network(layers, problems[i]['Input'][1], True)

            layers[-1][0].reset()


def train(layers, problems):
    """
    Train network

    Args:
    layers: list of tuples of the form (layer, method)
    problems: list if training problems
    """

    epochs = 4
    for _ in range(epochs):
        #for i in range(len(problems)):
        for i in np.random.permutation(len(problems)):
            # run the input windows through the network
            run_through_network(layers, problems[i]['Input'][0], True)
            run_through_network(layers, problems[i]['Input'][1], True)
            run_through_network(layers, problems[i]['Input'][2], True)

            # run the correct answer through the network
            res_idx = problems[i]['Attributes']['result'] - 1
            run_through_network(layers, problems[i]['Output'][res_idx], True)

            layers[-1][0].reset()


def test(layers, problems):
    """
    Test network

    Args:
    layers: list of tuples of the form (layer, method)
    problems: list of training problems
    """

    # list that contains the confidence for every correct prediction
    correct_predictions = np.zeros(len(problems))
    num_correct_predictions = 0

    for i in range(len(problems)):
        # run the input through the network
        for j in range(3):
            run_through_network(layers, problems[i]['Input'][j], False)

        # get predicted state
        predict = layers[-1][0].predict(1)

        # correct answer
        res_idx = problems[i]['Attributes']['result'] - 1

        # compare every output window with the predicted state
        matches = np.zeros(len(problems[i]['Output']))
        for j in range(len(problems[i]['Output'])):
            # run the window through all but the last layer
            last_input = run_through_network(layers[:-1], problems[i]['Output'][j], False)

            # compare the result with the predicted state
            matches[j] = np.sum(last_input * predict) / (np.sum(last_input) + np.sum(predict))

        #print(problems[i]['Attributes']['title'], matches)
        #vote = np.argmax(matches)

        # if the precition was correct add it to the correct_predictions list
        vote = np.argmax(matches)
        print(problems[i]['Attributes']['title'], vote, res_idx)
        if vote == res_idx and np.max(matches) > 0.0:
            correct_predictions[i] = np.max(matches)
            num_correct_predictions += 1

        # run the precited window through the layers and reset the temporal memory layer
        run_through_network(layers, problems[i]['Output'][vote], False)
        layers[-1][0].reset()

        '''
         # write output
        with open('output/htm/two_rows/' + str(i), 'w+') as f:
            for win in problems[i]['Input']:
                size = int(len(win) ** 0.5)
                for line in win.reshape((size, size)):
                    f.write(''.join([str(x) for x in line]) + '\n')
                f.write('\n')
            m = np.mean(predict)
            for line in predict.reshape((size, size)):
                f.write(''.join([str(int(x >= m)) for x in line]) + '\n')
            f.write('\n')
        '''


    print(correct_predictions)
    print(num_correct_predictions, len(correct_predictions))
    print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


def test_2(layers, problems):
    correct_predictions = np.zeros(len(problems))
    num_correct_predictions = 0
    windows = parse_images.get_windows(problems)


    print("Run windows")

    for _ in range(2):
        for i in np.random.permutation(len(windows)):
            run_through_network(layers, windows[i], True)
            run_through_network(layers, windows[i], True)
            layers[-1][0].reset()

    print("Run problems")
    for _ in range(5):
        for i in np.random.permutation(len(problems)):
            run_through_network(layers, problems[i]['Input'][0], True)
            run_through_network(layers, problems[i]['Input'][1], True)
            layers[-1][0].reset()

    print("Test")
    for i in range(len(problems)):
        for _ in range(10):
            run_through_network(layers, problems[i]['Input'][0], True)
            run_through_network(layers, problems[i]['Input'][1], True)
            layers[-1][0].reset()

        run_through_network(layers, problems[i]['Input'][2], False)
        predict = layers[-1][0].predict(1)
        # correct answer
        res_idx = problems[i]['Attributes']['result'] - 1

        matches = np.zeros(len(problems[i]['Output']))
        for j in range(len(problems[i]['Output'])):
            # run the window through all but the last layer
            last_input = run_through_network(layers[:-1], problems[i]['Output'][j], False)
            # compare the result with the predicted state

            matches[j] = np.sum(last_input * predict) / (np.sum(last_input) + np.sum(predict))
            '''
            print np.sum(last_input * predict), matches[j],
            if j == res_idx:
                print "*"
            else:
                print ""
            '''

        vote = np.argmax(matches)
        print(problems[i]['Attributes']['title'], vote, res_idx)

        # if the precition was correct add it to the correct_predictions list
        if vote == res_idx and np.max(matches) > 0.0:
            correct_predictions[i] = np.max(matches)
            num_correct_predictions += 1

        run_through_network(layers, problems[i]['Output'][vote], False)
        layers[-1][0].reset()

        '''
        # write output
        with open('output/htm/one_row/' + str(i), 'w+') as f:
            for win in problems[i]['Input']:
                size = int(len(win) ** 0.5)
                for line in win.reshape((size, size)):
                    f.write(''.join([str(x) for x in line]) + '\n')
                f.write('\n')
            m = np.mean(predict)
            for line in predict.reshape((size, size)):
                f.write(''.join([str(int(x >= m)) for x in line]) + '\n')
            f.write('\n')
        '''

    print(correct_predictions)
    print(num_correct_predictions, len(correct_predictions))
    print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


def run(layers, problems):
    #train_first_row(layers, problems)
    #train(layers, problems)
    #test(layers, problems)
    test_2(layers, problems)


def run_imgs():
    folder_name = 'Data/Problems'
    problems = parse_images.get_problems(folder_name)
    for problem in problems:
        problem['Input'] = problem['Input'].reshape((3, -1))
        problem['Output'] = problem['Output'].reshape((6, -1))

    # dimenstions
    num_colls1 = len(problems[0]['Input'][0])
    '''
    num_colls2 = num_colls1 / 2
    num_colls3 = num_colls2 / 2

    # layers

    sp1 = SpatialPooler(inputDimensions=(num_colls1, ),
                        columnDimensions=(num_colls2, ),
                        numActiveColumnsPerInhArea=-1,
                        localAreaDensity=0.05)

    sp2 = SpatialPooler(inputDimensions=(num_colls2, ),
                        columnDimensions=(num_colls3, ),
                        numActiveColumnsPerInhArea=-1,
                        localAreaDensity=0.05)

    '''
    bckTM = BTM(numberOfCols=num_colls1, cellsPerColumn=10,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=10, newSynapseCount=10,
                activationThreshold=10,
                pamLength=10)


    #layers = [(sp1, sp_compute),
    #          (sp2, sp_compute),
    #          (bckTM, bk_tm_compute)]

    layers = [(bckTM, bk_tm_compute)]


    run(layers, problems)


def run_sdrs():
    folder_name = 'Data/Problems_sdr'
    problems = parse_images.get_problems(folder_name)

    for problem in problems:
        assert problem['SDRs'].shape == (9, 32, 32)
        problem['Input'] = problem['SDRs'][:3].reshape((3, -1))
        problem['Output'] = problem['SDRs'][3:].reshape((6, -1))

    num_colls = len(problems[0]['Input'][0])
    bckTM = BTM(numberOfCols=num_colls, cellsPerColumn=15,
            initialPerm=0.5, connectedPerm=0.5,
            minThreshold=10, newSynapseCount=10,
            activationThreshold=10,
            pamLength=10)

    layers = [(bckTM, bk_tm_compute)]
    run(layers, problems)


def run_symbolic():
    folder_name = 'Data/Problems_txt'
    problems = read_symbolic_problems.get_problems(folder_name)

    num_colls = len(problems[0]['Input'][0])
    bckTM = BTM(numberOfCols=num_colls, cellsPerColumn=15,
            initialPerm=0.5, connectedPerm=0.5,
            minThreshold=10, newSynapseCount=10,
            activationThreshold=10,
            pamLength=10)

    layers = [(bckTM, bk_tm_compute)]
    run(layers, problems)


def main(args):
    if args.data == 'imgs':
        run_imgs()

    elif args.data == 'sdrs':
        run_sdrs()

    elif args.data == 'symbolic':
        run_symbolic()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default = 'imgs')
    parser.add_argument("--experiment", type=str, default = 'solve')
    args = parser.parse_args()

    main(args)
