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


def train(layers, problems):
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
            for j in range(3):
                last_input = problems[i]['Input'][j]
                for (layer, compute_method) in layers:
                    last_input = compute_method(layer, last_input, True)

            # run the correct answer through the network
            res_idx = problems[i]['Attributes']['result'] - 1
            last_input = problems[i]['Output'][res_idx]
            for (layer, compute_method) in layers:
                last_input = compute_method(layer, last_input, True)


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
            last_input = problems[i]['Input'][j]
            for (layer, compute_method) in layers:
                last_input = compute_method(layer, last_input, False)

        # get predicted state
        predict = layers[-1][0].predict(1)

        # correct answer
        res_idx = problems[i]['Attributes']['result'] - 1

        # compare every output window with the predicted state
        matches = np.zeros(len(problems[i]['Output']))
        for j in range(len(problems[i]['Output'])):
            # run the window through all but the last layer
            last_input = problems[i]['Output'][j]
            for (layer, compute_method) in layers[:-1]:
                last_input = compute_method(layer, last_input, False)
            # compare the result with the predicted state
            matches[j] = np.sum(last_input * predict)

        # if the precition was correct add it to the correct_predictions list
        if np.argmax(matches) == res_idx and np.max(matches) > 0.0:
            correct_predictions[i] = np.max(matches)
            num_correct_predictions += 1

    print(correct_predictions)
    print(num_correct_predictions, len(correct_predictions))
    print(np.sum(correct_predictions) * 100.0 / len(correct_predictions))


def run(problems):
    """
    Define, train and test a network of spatial pooler and temporal memory layers
    """

    # dimenstions
    numColls1 = len(problems[0]['Input'][0])
    numColls2 = numColls1 / 2
    numColls3 = numColls2 / 2
    numColls4 = numColls3 / 2

    # layers
    sp1 = SpatialPooler(inputDimensions=(numColls1, ),
                        columnDimensions=(numColls2, ),
                        numActiveColumnsPerInhArea=-1,
                        localAreaDensity=0.05)
    sp2 = SpatialPooler(inputDimensions=(numColls2, ),
                        columnDimensions=(numColls3, ),
                        numActiveColumnsPerInhArea=-1,
                        localAreaDensity=0.05)
    '''
    sp3 = SpatialPooler(inputDimensions = (numColls3, ), columnDimensions = (numColls4, ),
                        numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
    sp4 = SpatialPooler(inputDimensions = (numColls3, ), columnDimensions = (numColls4, ),
                        numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
    '''
    #minThr = int(numColls1 * 0.2 / 100)
    #actThr = int(numColls1 * 0.2 / 100)
    #minThr = int(numColls1 ** 0.33 / 2)
    #actThr = int(numColls1 ** 0.33 / 2)
    minThr = 2
    actThr = 2
    print(minThr, actThr)
    bckTM = BTM(numberOfCols=numColls1, cellsPerColumn=20,
                initialPerm=0.5, connectedPerm=0.5,
                minThreshold=minThr, newSynapseCount=10,
                activationThreshold=actThr,
                pamLength=10)
    '''
    layers = [(sp1, sp_compute),
              (sp2, sp_compute),
              (bckTM, bk_tm_compute)]
    '''
    layers = [(bckTM, bk_tm_compute)]
    train(layers, problems)
    test(layers, problems)


def main(args):
    if args.data == 'imgs':
        folder_name = 'Data/Problems'
        problems = parse_images.get_problems(folder_name)

        for problem in problems:
            problem['Input'] = problem['Input'].reshape((3, -1))
            problem['Output'] = problem['Output'].reshape((6, -1))
        run(problems)

    elif args.data == 'sdrs':
        folder_name = 'Data/Problems_sdr'
        problems = parse_images.get_problems(folder_name)

        for problem in problems:
            assert problem['SDRs'].shape == (9, 32, 32)
            problem['Input'] = problem['SDRs'][:3].reshape((3, -1))
            problem['Output'] = problem['SDRs'][3:].reshape((6, -1))
        run(problems)

    elif args.data == 'symbolic':
        folder_name = 'Data/Problems_txt'
        problems = read_symbolic_problems.get_problems(folder_name)
        run(problems)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default = 'imgs')
    args = parser.parse_args()

    main(args)
