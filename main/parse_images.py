import numpy as np
import os
import png
import re
from argparse import ArgumentParser


def identify_colors(img_struct, img):
    """
    Prepocess the image in order to identify 3 colors: white, black, blue

    Args:
    img_struct: dictionary containing image attributes
    img: matrix of pixels

    Returns:
    img: matrix of integers with the same size as the input image but containing only:
            -> 0 for white
            -> 1 for black
            -> 2 for blue
    """

    if img_struct[3]['planes'] == 1:
        palette = [2 if (p[2] > p[1] and p[2] > p[0] and sum(p) / 3 < 130 and sum(p) / 3 > 135)
                    else (1 - sum(p) / (3 * 255.0))
                   for p in img_struct[3]['palette']]
        img = [[palette[x] for x in l] for l in img]
    else:
        img = [[2 if l[i + 2] > l[i + 1] and l[i + 2] > l[i] and sum(l[i : i + 3]) / 3 > 80 and sum(l[i : i + 3]) / 3 <= 230 else
                1 - int(round((l[i] + l[i + 1] + l[i + 2]) / (3 * 255.0)))
                    for i in range(0, len(l), 3)]
                for l in img]

    for _ in range(1):
        count_2s = [[0] * len(l) for l in img]
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] == 2:
                    for (a, b) in [(a, b) for a in [0, 1, -1] for b in [0, 1, -1]]:
                        if i + a >= 0 and \
                           j + b >= 0 and \
                           i + a < len(img) and \
                           j + b < len(img[i]):
                            count_2s[i + a][j + b] += 1

        count_2s = [[min([1, x / 3]) for x in l] for l in count_2s]

        img = [[count_2s[i][j] * 2 if img[i][j] == 2 else img[i][j]
                for j in range(len(img[i]))]
                for i in range(len(img))]

    return img


def split_into_windows(img):
    """
    Extract the input and output windows from the image of a problem

    Args: receives a matrix containing 0, 1, 2 for white, black, blue respectively

    Returns: a list of windows
    """
    size = (len(img), len(img[0]))

    # find the start of first window
    x = 0
    y = 0
    while img[x][y] != 2:
        y += 1
        if y== len(img[0]):
            y = 0
            x += 1

    # find the end of the first window and compute the window size
    window_size = (x, y)
    while img[x][window_size[1]] == 2:
        x += 1
    while img[window_size[0]][y] == 2:
        y += 1
    window_size = (x - window_size[0], y - window_size[1])


    x = 1
    y = 1
    windows = []
    found = 0
    while x < size[0] and y < size[1]:
        # go through the matrix and find window corners
        if img[x][y] == 2 or \
        (found and img[x + 1][y] == 2):
            found = 1

            # extract the window and add it to the list
            new_window = [img[x + i][y: y + window_size[1]] for i in range(window_size[0])]
            windows.append(np.array(new_window))

            # move beyond the window
            y += window_size[1]
            if y >= size[1]:
                y = 0
                x += window_size[0]

        # if no corner is found cotinue to move by one pixel at a time
        y += 1
        if y >= size[1]:
            y = 0
            x += 1 + found * window_size[0]
            found = 0

    # for all windows change blue to white (2 -> 0)
    pix_dict = {0:'0', 1:'1', 2:'0'}
    for i in range(len(windows)):
        windows[i] = [[pix_dict[x] for x in line] for line in windows[i]]
    return np.array(windows)


def parse_problem(problem, png_file, txt_file):
    """
    Parse information from the text and image version of the problem
    in order to obtain the correct data structure to work on

    Args:
    problem: empty dictionary that will store the result
    png_file: path to the image of the problem
    txt_file: path to the text of the problem
    """
    # read the text file in order to get the attributes of the poblem
    problem['Attributes'] = {}
    with open(txt_file) as f:
        problem['Attributes']['title' ] = f.readline().rstrip()
        problem['Attributes']['type'  ] = f.readline().rstrip()
        problem['Attributes']['result'] = int(f.readline().rstrip())

    # open the image and separate the windows
    f = open(png_file)
    r=png.Reader(file=f)
    img_struct = r.read()
    img = list(img_struct[2])
    img = identify_colors(img_struct, img)
    f.close()

    windows = split_into_windows(img)
    problem['Attributes']['window_size'] = windows[0].shape

    # store the windows in the appropriate list
    # ignore the 4th windows as it is a question mark
    problem['Input']  = windows[:3]
    problem['Output'] = windows[4:]


def write_problem(problem, out_file):
    """
    Write the information of one problem to a file

    Args:
    problem: dictionary that contains the keys 'Attributes', 'Input', 'Output'
        and fully describes a problem; it can also contain the SRDs created for every window
    out_file: name of the output file
    """

    # check sizes
    window_size = list(problem['Attributes']['window_size'])
    assert problem['Input'].shape == tuple([3] + window_size), 'Wrong input wondow size'
    assert problem['Output'].shape == tuple([6] + window_size), 'Wrong output wondow size'

    with open(out_file, 'w+') as f:
        # write attributes
        f.write('== Attributes ==\n')
        f.write('title=' + problem['Attributes']['title'] + '\n')
        f.write('type= '+ problem['Attributes']['type'] + '\n')
        f.write('result=' + str(problem['Attributes']['result']) + '\n')
        f.write('window_size=' + str(problem['Attributes']['window_size']) + '\n\n')

        # write the input windows
        f.write('== Input ==' + '\n')
        for window in problem['Input']:
            for line in window:
                f.write(''.join([str(x) for x in line]))
                f.write('\n')
            f.write('\n')

        # write the output windows
        f.write('== Output ==' + '\n')
        for window in problem['Output']:
            for line in window:
                f.write(''.join([str(x) for x in line]))
                f.write('\n')
            f.write('\n')

        # write the SDRs (if they exist)
        if 'SDRs' in problem:
            f.write('== SDRs ==' + '\n')
            for window in problem['SDRs']:
                for line in window:
                    f.write(''.join([str(x) for x in line]))
                    f.write('\n')
                f.write('\n')


def write_sdr_csv(problem, out_file):
    """
    Create a CSV file containing the information of one problem
    """
    with open(out_file, 'w+') as f:
        f.write('Problem\n')
        f.write('sdr\n')

        for window in problem['Input']:
            f.write(''.join([str(x) for line in window for x in line]))
            f.write('\n\n')

        window = problem['Output'][problem['Attributes']['result'] - 1]
        for line in window:
            f.write(''.join([str(x) for x in line]))
        f.write('\n')


def write_problems_with_sdrs(input_folder, output_folder):
    """
    Read problems from the input folder
    Encode all the windows using an auto-encoder
    Write the problems with the added encoded windows to the output_folder
    """

    import encoder

    # get all the windows from the problems in the folder
    # in order to train the encoder
    input_windows = get_windows(input_folder)
    (num_windows, height, width) = input_windows.shape

    # initialize and train the encoder
    enc = encoder.Encoder(height * width)
    errs = enc.train(input_windows)

    # get all the problems from the folder in order to create the SDRs
    problems = get_problems(input_folder)
    for i in range(len(problems)):
        # write the original form of the problem in the new file
        out_file = output_folder + '/' + str(i) + '.txt'
        write_problem(problems[i], out_file)

        # create and write to the new file SDRs for all the windows in the problem
        with open(out_file, 'a') as f:
            f.write('== SDRs ==\n')
            # for every window in the problem
            for win in np.concatenate((problems[i]['Input'], problems[i]['Output'])):
                # create SDR by encoding the window and then converting the list of float
                # values to 1s and 0s
                encoded = enc.encode(win.reshape((1, height, width, 1)))
                m = np.mean(encoded)
                encoded = (encoded >= m).astype(int)

                # write the SDR to the file
                size = int(len(encoded[0]) ** 0.5)
                for line in encoded.reshape((size, size)):
                    f.write(''.join([str(x) for x in line]) + '\n')
                f.write('\n')

                print('Sparsity ', float(np.sum(encoded) * 100) / encoded.shape[1])

                # decode the SDR
                decoded = enc.decode(encoded)
                m = np.mean(decoded)
                decoded = (decoded >= m).astype(int)

                # print the similarity between the input and the output
                print(np.sum(win == decoded.reshape(win.shape)) * 100.0 / (height * width))
                print("---------------")


def get_problems(folder_name):
    """
    Get all the problems from a given folder

    Returns: list of dictionaries, each describing one problem
    """

    problems = []
    for i in range(len(os.listdir(folder_name))):
        # for every file in the folder
        file_name = '%d.txt' % i
        problem = {}
        with open(os.path.join(folder_name, file_name)) as f:
            lines = f.read().split('\n')

            # the first line should be '== Attributes =='
            key = re.match('== (.*) ==', lines[0]).group(1)

            # parse the attributes of the problem
            problem[key] = {}
            index = 1   # index of the current line
            while lines[index] != '':
                m = re.match('(.*)=(.*)', lines[index])
                problem[key][m.group(1)] = m.group(2).strip()
                index += 1

            # cast certain attributes to their appropriate types
            problem['Attributes']['result'] = int(problem['Attributes']['result'])
            m = re.match('\((.*), (.*)\)', problem['Attributes']['window_size'])
            problem['Attributes']['window_size'] = (int(m.group(1)), int(m.group(1)))

            # read windows
            while index < len(lines) - 1:
                index += 1
                # match the line separating groups of windows (Input, Output, SDRs)
                key = re.match('== (.*) ==', lines[index]).group(1)
                # add the new group to the problem
                problem[key] = []

                index += 1
                while index < len(lines) - 1:
                    # read window line by line
                    new_window = []
                    while lines[index] != '':
                        new_window += [[int(x) for x in lines[index]]]
                        index += 1

                    # add window to the current group
                    problem[key].append(np.array(new_window))

                    # if the end of the file or the start of a new group was reached
                    # stop reading windows
                    if index < len(lines)-1 and re.match('== (.*) ==', lines[index + 1]):
                        break
                    index += 1
                problem[key] = np.array(problem[key])
        problems += [problem]

    return problems


def get_windows(folder_name):
    """
    Get all the windows out of all the problems in the given folder
    """
    problems = get_problems(folder_name)

    windows = np.append(problems[0]['Input'],
                        problems[0]['Output'],
                        axis=0)
    for problem in problems[1:]:
        windows = np.append(windows, problem['Input'], axis=0)
        windows = np.append(windows, problem['Output'], axis=0)

    return windows


if __name__ == '__main__':
    # parse argumets
    parser = ArgumentParser()
    parser.add_argument("--parse", dest="parse", action="store_true")
    parser.add_argument("--sdrs", dest="sdrs", action="store_true")
    args = parser.parse_args()

    # define problem folders
    problem_dir = '../Problem_images/resized'
    problem_txt_dir = '../Problems_txt'
    output_dir = '../Problems_try'

    if args.parse:
        problems = []
        index = 0

        folder_name = '../Problem_images/resized'
        for file_name in os.listdir(folder_name):
            problem = {}

            txt_file = os.path.join(problem_txt_dir, file_name.split('.')[0] + '.txt')
            png_file = os.path.join(folder_name, file_name)
            out_file = os.path.join(output_dir, str(index) + '.txt')
            csv_file = os.path.join("../CSV_Problems", str(index) + '.csv')
            index += 1

            parse_problem(problem, png_file, txt_file)
            write_problem(problem, out_file)
            write_sdr_csv(problem, csv_file)

    if args.sdrs:
        write_problems_with_sdrs(output_dir, '../Problems_sdr')

