import os
import re


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


def get_problems():
    """
    Load texr problems

    Args:

    Returns:
    problems: list of dictionaries describing each problem
    uniqie_attributes: dictionary describing a tree of attributes; it contains
                the structure of all possibile attributes in the list of problems
    """
    problem_dir = "../Problems_txt"
    problems = []

    for file in os.listdir(problem_dir):
        new_file = os.path.join(problem_dir, file)
        with open(new_file) as f:
            new_problem = {}
            new_problem['title'] = f.readline().rstrip()
            new_problem['type'] = f.readline().rstrip()
            new_problem['result'] = int(f.readline().rstrip())
            lines = f.readlines()
            new_problem['content'], _ = read_content(lines, 0, 0)
            problems.append(new_problem)

    return problems, get_unique_attributes(problems)
