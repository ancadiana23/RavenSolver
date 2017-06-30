import os
import re
import numpy as np


def unique_attributes(problem, problem_attributes):
    """
    Extract all the attributes in a problem and add them to the problem attributes dictionary.
    By running this function on all problems the problem_attributes dictionary
    will describe the structure of a tree that encodes all the problems.

    Args:
    problem: a dictionary describing one problem
    problem_attributes: a dictionary describing the current tree to which the
        current problem will be added
    """

    for key in problem:
        if key not in problem_attributes:
            # if the current node of the problem is not in the resulting tree
            # then check if the node is a leaf or not and add the appropaite
            # data structure to the resulting tree
            if type(problem[key]) is dict:
                problem_attributes[key] = {}
            else:
                problem_attributes[key] = set([])

        if type(problem_attributes[key]) is dict:
            # parse the next level of this branch
            unique_attributes(problem[key], problem_attributes[key])
        else:
            # when a leaf in the problem is reached add the value of the leaf
            # to a list in the problem_attributes
            problem_attributes[key].add(problem[key])


def read_content(lines, index, level):
    """
    Recursive function that parses the lines on a problem file.

    Args:
    lines: lines of the file
    index: index of the current line
    level: the current level in the content dictionary
    """

    content = {}
    while index < len(lines):
        line = lines[index].rstrip()
        new_level = 0
        while line[0] == '\t':
            # count the tabs at the beginning the of the line to find the depth
            #   (level) of the element
            line = line[1:]
            new_level += 1

        if new_level == level:
            # if the level of the content is equal to the level of the
            # element in the line, add the element to the dictionary
            # otherwise make a new level in the content
            if ':' in line:
                [key, value] = re.split(':', line)
                content[key] = value
                index += 1
            else:
                content[line], index = read_content(lines, index + 1, level + 1)
        elif new_level < level:
            break

    return content, index


def get_unique_attributes(problems):
    """
    Recursive functiona that creates a tree of all the possible attributes in the
    list of problems

    Args:
    problems: list of dictioaries describing the problems

    Returns:
    attributes: dictionary of all the possible attributes in the list of problems
    """
    problem_attributes = {}
    for problem in problems:
        for key in problem['content']:
            unique_attributes(problem['content'][key], problem_attributes)

    return problem_attributes


def read_problems(problem_dir):
    """
    Load texr problems

    Args:
    problem_dir: the path to the folder of problems

    Returns:
    problems: list of dictionaries describing each problem
    """
    problems = []

    for file in os.listdir(problem_dir):
        new_file = os.path.join(problem_dir, file)
        with open(new_file) as f:
            new_problem = {'Attributes': {}}
            new_problem['Attributes']['title'] = f.readline().rstrip()
            new_problem['Attributes']['type'] = f.readline().rstrip()
            new_problem['Attributes']['result'] = int(f.readline().rstrip())
            lines = f.readlines()
            new_problem['content'], _ = read_content(lines, 0, 0)
            problems.append(new_problem)

    return problems


def create_SDR(window, problem_attributes, SDR, match=True):
    """
    Create the SDR that encodes a window.

    Args:
    window: dictionary describing a window
    problem_attributes: dictionary that encodes all the possibile attributes in all the windows
    SDR: list that will contain the result
    match: boolean that states of the current branch in the
        problem_attributes branch was previosly mqatched in the window;
        used to add 0s for all other possbile attributes

    Returns:
    a list of 0s and 1s that encodes the input window
    """

    if match:
        # if the previous attributes on the current branched are present in the window
        if type(problem_attributes) is dict:
            for key in sorted(problem_attributes.keys()):
                # for every param on the current level of the current branch
                # check if it is in the window
                if key in window:
                    # if it is, continue down its branch
                    create_SDR(window[key], problem_attributes[key], SDR, match)
                else:
                    # if it is not, add 0s for all the nodes on its branch
                    create_SDR({}, problem_attributes[key], SDR, False)
        else:
            # if a leaf was reached add 1 for the attribute on that leaf
            # and 0 for all the other possibilities
            attributes = sorted(list(problem_attributes))
            SDR.extend(int(x == window) for x in attributes)
    else:
        # if the current branch was not previously matched then just add 0s
        # in order to preserve a consistent structure for all SDRs
        if type(problem_attributes) is dict:
            for key in problem_attributes:
                create_SDR({}, problem_attributes[key], SDR, match)
        else:
            SDR.extend([0] * len(problem_attributes))


def get_problems(problem_dir):
    """
    Get the pre-processed  problems

    Args:
    problem_dir: path to the folder of problems

    Returns: a list of dictionaries describing the problems
    """

    problems = read_problems(problem_dir)
    problem_attributes = get_unique_attributes(problems)
    SDRs = []

    for problem in problems:
        problem['Input'] = []
        problem['Output'] = []
        aux = []
        for (x, y) in [('A', 'Input'), ('1', 'Output')]:
            # for every input window, desbribed by letter A-C,
            # and for every output window, described by numbers 1-6,
            # create an SDR
            index = ord(x)
            while chr(index) in problem['content']:
                new_SDR = []
                create_SDR(problem['content'][chr(index)], problem_attributes, new_SDR)
                SDR = []
                for x in new_SDR:
                    for _ in range(5):
                        SDR += [x]
                problem[y].append(np.array(SDR))
                aux += [np.sum(SDR)]
                index += 1
        problem['Input'] = np.array(problem['Input'])
        problem['Output'] = np.array(problem['Output'])

    return problems
