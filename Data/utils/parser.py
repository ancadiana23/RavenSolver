from argparse import ArgumentParser
from yaml import load
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from scipy.misc import imrotate
from scipy.ndimage import rotate,zoom

import os
import re
import numpy as np


def parse_window(window, height, width, constants):
    result = [0 for _ in range(height) for _ in range(width)]
    for figure in window:
        new_fig = constants['shapes'][window[figure]['shape']]
        size = (int(pow(len(new_fig), 0.5)), int(pow(len(new_fig),0.5)))

        new_fig = np.array(new_fig)
        new_fig = np.reshape(new_fig, size)

        new_fig = rotate(new_fig, window[figure].get('rotation', 0))

        fill = [0 for _ in range(size[0]) for _ in range(size[1])]
        for pattern in window[figure].get('fill', ['none']):
            fill = [min(1, fill[i] + constants['fills'][pattern][i]) for i in range(len(fill))]
        for i in range(size[0]):
            for j in range(size[1]):
                if new_fig[i][j] == 2:
                    new_fig[i][j] = fill[i * size[1] + j]

        resize_multiplier = window[figure].get('size', 1) / 3
        new_fig = zoom(new_fig, resize_multiplier)

        #new_fig = list(np.reshape(new_fig, size[0] * size[1]))

        #new_fig = [fill[i] if new_fig[i] == 2 else new_fig[i]  for i in range(len(new_fig))]
        #new_fig = [x * 255 for x in new_fig]

        #img = Image.new('L', size)
        #img.putdata(new_fig)

        #img = img.rotate(window[figure].get('rotation', 0))

        #resize_multiplier = window[figure].get('size', 1) / 3
        #size = (int(size[0] * resize_multiplier), int(size[1] * resize_multiplier))
        #img = img.resize(size)

        #new_fig = list(img.getdata())

        #new_fig = np.array([[new_fig[i * size[1] + j] != 0 for j in range(size[1])] for i in range(size[0])])
        position_dict = {'left': 0, 'center': 1, 'right': 2}

        padding_line = int((width - size[0]) / 2.0)
        padding_column = int((height - size[1]) / 2.0)
        pos = (position_dict[window[figure]['line']] * padding_line,
               position_dict[window[figure]['column']] * padding_column)
        for i in range(len(new_fig)):
            start = (pos[0] + i) * width + pos[1]
            end = start + len(new_fig[i])
            result[start : end] += new_fig[i]
    return result


def write_problem(problem):
    file_name = "./raw/" + problem["Attributes"]['title'] + '.txt'
    with open(file_name, 'w+') as f:
        # write attributes
        f.write('== Attributes ==\n')
        f.write('title=' + problem['Attributes']['title'] + '\n')
        f.write('type= '+ problem['Attributes']['type'] + '\n')
        f.write('result=' + str(problem['Attributes']['result']) + '\n')
        f.write('window_size=' + str(problem['Attributes']['window_size']) + '\n\n')

        height = problem['Attributes']['window_size'][0]
        width = problem['Attributes']['window_size'][1]
        # write the input windows
        f.write('== Input ==' + '\n')
        for window in problem['Input']:
            for i in range(height):
                f.write(''.join([str(x) for x in window[i * width : (i + 1) * width]]))
                f.write('\n')
            f.write('\n')

        # write the output windows
        f.write('== Output ==' + '\n')
        for window in problem['Output']:
            for i in range(height):
                f.write(''.join([str(x) for x in window[i * width : (i + 1) * width]]))
                f.write('\n')
            f.write('\n')

        # write the SDRs (if they exist)
        if 'SDRs' in problem:
            f.write('== SDRs ==' + '\n')
            for window in problem['SDRs']:
                for i in range(height):
                    f.write(''.join([str(x) for x in window[i * width : (i + 1) * width]]))
                    f.write('\n')
                f.write('\n')


def write_image(problem):
    height = 512
    width = 1024
    list_img = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    win_len = int(pow(len(problem['Input'][0]), 0.5))
    frame_len = win_len + 2
    start_pos_y = int((height - (5 * frame_len)) / 2)
    problem['Input'] += [[0 for _ in range(len(problem['Input'][0]))]]
    for (set, start, end) in [('Input', 0, 2), ('Input', 2, 4),  ('Output', 0, 6)]:
        num_win = len(problem[set][start : end])
        start_pos_x = int((width - ((2 * num_win - 1) * frame_len)) / 2)
        for i in range(start, end):
            start_x = start_pos_x + (i - start) * int(2 * frame_len)
            list_img[start_pos_y][start_x : start_x + frame_len] = [(0, 0, 70) for _ in range(frame_len)]
            for line in range(1, win_len):
                list_img[start_pos_y + line][start_x] = (0, 0, 70)
                list_img[start_pos_y + line][start_x + frame_len - 1] = (0, 0, 70)
                list_img[start_pos_y + line][start_x + 1 : start_x + win_len + 1] = [tuple([int((1 - x) * 255) for _ in range(3)]) for x in problem[set][i][line * win_len : (line + 1) * win_len]]

            list_img[start_pos_y + frame_len - 2][start_x : start_x + frame_len] = [(0, 0, 70) for _ in range(frame_len)]
        start_pos_y += int(2 * frame_len)

    img = Image.new('RGB', (width, height))
    img.putdata([x for line in list_img for x in line])

    text_size = 20
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', text_size)
    line = 0
    for attribute in problem['Attributes']:
        draw.text((0, line), attribute + ' = ' + str(problem['Attributes'][attribute]) ,(0, 0, 70), font=font)
        line += text_size

    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 75)
    draw.text((565, 217), '?' ,(0, 0, 70), font=font)
    img.save("./images/" + problem["Attributes"]['title'] + '.png')


def read_constants(file_name):
    constants = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        size = int(lines[0])
        i = 2
        while i < len(lines):
            name = lines[i].rstrip()
            i += 1
            constants[name] = [int(x) for j in range(size) for x in lines[i + j].rstrip()]
            i += size + 2
    return constants


def main(args):
    constants = {}
    constants['shapes'] = read_constants('constants.txt')
    constants['fills'] = read_constants('fills.txt')

    for file in os.listdir(args.input_folder):
        file_name = os.path.join(args.input_folder, file)
        with open(file_name) as file:
            print(file_name)
            text = file.read()
            data = load(text)
            m = re.match("\((.*), (.*)\)", data["Attributes"]["window_size"])
            data["Attributes"]["window_size"] = (int(m.group(1)), int(m.group(2)))

            for i in range(0, len(data["Input"])):
                data["Input"][i] = parse_window(data["Input"][i],
                                                data["Attributes"]["window_size"][0],
                                                data["Attributes"]["window_size"][1],
                                                constants)

            for i in range(0, len(data["Output"])):
                data["Output"][i] = parse_window(data["Output"][i],
                                                 data["Attributes"]["window_size"][0],
                                                 data["Attributes"]["window_size"][1],
                                                 constants)

            write_problem(data)
            write_image(data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, default = 'desc')
    parser.add_argument("--raw_folder", type=str, default = 'raw')
    parser.add_argument("--img_folder", type=str, default = 'images')

    args = parser.parse_args()

    main(args)