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


def write_windows(file_name, problem, predict):
    with open(file_name, 'w+') as f:
        for win in problem['Input']:
            size = int(len(win) ** 0.5)
            for line in win.reshape((size, size)):
                f.write(''.join([str(x) for x in line]) + '\n')
            f.write('\n')
        m = np.mean(predict)
        for line in predict.reshape((size, size)):
            f.write(''.join([str(int(x >= m)) for x in line]) + '\n')
        f.write('\n')


def run_through_network(layers, input_data, learn=True):
    last_input = input_data
    for (layer, compute_method) in layers:
        last_input = compute_method(layer, last_input, learn)
    return last_input


def train_input(layers, problems, indexes):
    """
    Train network

    Args:
    layers: list of tuples of the form (layer, method)
    problems: list if training problems
    """
    epochs = 3
    for _ in range(epochs):
        #for i in range(len(problems)):
        for i in np.random.permutation(len(problems)):
            # run the input windows through the network
            for j in indexes:
                run_through_network(layers, problems[i]['Input'][j], True)

            layers[-1][0].reset()


def train_windows(layers, problems, sequence_len):
    windows = parse_images.get_windows(problems)

    for _ in range(5):
        for i in np.random.permutation(len(windows)):
            for _ in range(sequence_len):
                run_through_network(layers, windows[i], True)
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


def test(layers, problems, write_output):
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

        # if the precition was correct add it to the correct_predictions list
        vote = np.argmax(matches)
        print(problems[i]['Attributes']['title'], vote, res_idx)
        if vote == res_idx and np.max(matches) > 0.0:
            correct_predictions[i] = np.max(matches)
            num_correct_predictions += 1

        # run the precited window through the layers and reset the temporal memory layer
        run_through_network(layers, problems[i]['Output'][vote], False)
        layers[-1][0].reset()

        if write_output:
            write_windows('./output/htm/4/' + str(i) + '.txt', problems[i], predict)

    print(correct_predictions)
    print(num_correct_predictions, len(correct_predictions))
    print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


def test_2_windows(layers, problems, write_output):
    correct_predictions = np.zeros(len(problems))
    num_correct_predictions = 0

    for i in range(len(problems)):
        for _ in range(20):
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


        vote = np.argmax(matches)
        print(problems[i]['Attributes']['title'], vote, res_idx)

        # if the precition was correct add it to the correct_predictions list
        if vote == res_idx and np.max(matches) > 0.0:
            correct_predictions[i] = np.max(matches)
            num_correct_predictions += 1

        run_through_network(layers, problems[i]['Output'][vote], False)
        layers[-1][0].reset()

        if write_output:
            write_windows('./output/htm/2/' + str(i) + '.txt', problems[i], predict)

    print(correct_predictions)
    print(num_correct_predictions, len(correct_predictions))
    print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


def init_imgs(write_output):
    folder_name = 'Data/Problems'
    problems = parse_images.get_problems(folder_name)

    for problem in problems:
        problem['Input'] = problem['Input'].reshape((3, -1))
        problem['Output'] = problem['Output'].reshape((6, -1))

    # dimenstions
    num_cols1 = len(problems[0]['Input'][0])
    tm_cols = num_cols1
    layers = []

    if not write_output:
        num_cols2 = num_cols1 / 2
        num_cols3 = num_cols2 / 2
        sp1 = SpatialPooler(inputDimensions=(num_cols1, ),
                            columnDimensions=(num_cols2, ),
                            numActiveColumnsPerInhArea=-1,
                            localAreaDensity=0.05)

        sp2 = SpatialPooler(inputDimensions=(num_cols2, ),
                            columnDimensions=(num_cols3, ),
                            numActiveColumnsPerInhArea=-1,
                            localAreaDensity=0.05)
        tm_cols = num_cols3
        layers = [(sp1, sp_compute),
                  (sp2, sp_compute)]


    bckTM = BTM(numberOfCols=tm_cols, cellsPerColumn=10,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=10, newSynapseCount=10,
                activationThreshold=10,
                pamLength=10)

    layers += [(bckTM, bk_tm_compute)]

    return (layers, problems)


def init_sdrs():
    folder_name = 'Data/Problems_sdr'
    problems = parse_images.get_problems(folder_name)

    for problem in problems:
        assert problem['SDRs'].shape == (9, 32, 32)
        problem['Input'] = problem['SDRs'][:3].reshape((3, -1))
        problem['Output'] = problem['SDRs'][3:].reshape((6, -1))

    num_cols = len(problems[0]['Input'][0])
    bckTM = BTM(numberOfCols=num_cols, cellsPerColumn=15,
            initialPerm=0.5, connectedPerm=0.5,
            minThreshold=10, newSynapseCount=10,
            activationThreshold=10,
            pamLength=10)

    layers = [(bckTM, bk_tm_compute)]

    return (layers, problems)


def init_symbolic():
    folder_name = 'Data/Problems_txt'
    problems = read_symbolic_problems.get_problems(folder_name)

    num_cols = len(problems[0]['Input'][0])
    bckTM = BTM(numberOfCols=num_cols, cellsPerColumn=15,
            initialPerm=0.5, connectedPerm=0.5,
            minThreshold=10, newSynapseCount=10,
            activationThreshold=10,
            pamLength=10)

    layers = [(bckTM, bk_tm_compute)]
    return (layers, problems)


def main(args):
    if args.data == 'imgs':
        (layers, problems) = init_imgs(args.write_pred)

    elif args.data == 'sdrs':
        (layers, problems) = init_sdrs()

    elif args.data == 'symbolic':
        (layers, problems) = init_symbolic()

    if args.ex == 'memorize':
        train(layers, problems)
        test(layers, problems, args.write_pred)

    elif args.ex == 'solve':
        if args.app =='4':
            train_windows(layers, problems, 4)
            train_input(layers, problems, [0, 1, 0, 1])
            test(layers, problems, args.write_pred)

        if args.app == '2':
            train_windows(layers, problems, 2)
            train_input(layers, problems, [0, 1])
            test_2_windows(layers, problems, args.write_pred)



if __name__ == "__main__":
    parser = ArgumentParser()
    # imgs, sdrs, symbolic
    parser.add_argument("--data", type=str, default = 'imgs')

    # experiment -> memorize, solve,
    parser.add_argument("--ex", type=str, default = 'solve')

    # approach -> 2, 4
    parser.add_argument("--app", type=str, default = '2')

    # write predicted window
    parser.add_argument("--w", dest="write_pred", action = "store_true")
    args = parser.parse_args()

    main(args)
