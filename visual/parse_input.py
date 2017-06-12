import numpy as np
import os
import png
import re


def identify_colors(img_struct, img):
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
	size = (len(img), len(img[0]))
	x = 0
	y = 0
	while img[x][y] != 2:
		y += 1
		if y== len(img[0]):
			y = 0
			x += 1
	
	window_size = (x, y)
	while img[x][window_size[1]] == 2:
		x += 1
	while img[window_size[0]][y] == 2:
		y += 1
	window_size = (x - window_size[0], y - window_size[1])
	problem['Attributes']['window_size'] = window_size
	
	x = 1
	y = 1
	windows = []
	found = 0
	while x < size[0] and y < size[1]:
		if img[x][y] == 2 or \
		(found and img[x + 1][y] == 2):
			found = 1
			new_window = [img[x + i][y: y + window_size[1]] for i in range(window_size[0])]
			windows.append(np.array(new_window))
			y += window_size[1]
			if y >= size[1]:
				y = 0
				x += window_size[0]

		y += 1
		if y >= size[1]:
			y = 0
			x += 1 + found * window_size[0]
			found = 0

	pix_dict = {0:'0', 1:'1', 2:'0'}
	for i in range(len(windows)):
		windows[i] = [[pix_dict[x] for x in line] for line in windows[i]]
	return np.array(windows)


def parse_problem(problem, png_file, txt_file):
	problem['Attributes'] = {}
	with open(txt_file) as f:
		problem['Attributes']['title' ] = f.readline().rstrip()
		problem['Attributes']['type'  ] = f.readline().rstrip()
		problem['Attributes']['result'] = int(f.readline().rstrip())

	f = open(png_file)
	r=png.Reader(file=f)
	img_struct = r.read()
	img = list(img_struct[2])
	img = identify_colors(img_struct, img)
	f.close()

	windows = split_into_windows(img)

	problem['Input']  = windows[:3]
	problem['Output'] = windows[4:]


def write_problem(problem, out_file):
	window_size = list(problem['Attributes']['window_size'])
	assert problem['Input'].shape == tuple([3] + window_size), 'Wrong input wondow size'
	assert problem['Output'].shape == tuple([6] + window_size), 'Wrong output wondow size'

	with open(out_file, 'w+') as f:
		f.write('== Attributes ==\n')
		f.write('title=' + problem['Attributes']['title'] + '\n')
		f.write('type= '+ problem['Attributes']['type'] + '\n')
		f.write('result=' + str(problem['Attributes']['result']) + '\n')
		f.write('window_size=' + str(problem['Attributes']['window_size']) + '\n\n')

		f.write('== Input ==' + '\n')
		for window in problem['Input']:
			for line in window:
				f.write(''.join([str(x) for x in line]))
				f.write('\n')
			f.write('\n')

		f.write('== Output ==' + '\n')
		for window in problem['Output']:
			for line in window:
				f.write(''.join([str(x) for x in line]))
				f.write('\n')
			f.write('\n')


def write_sdr_csv(problem, out_file):
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
    import encoder

    input_windows = get_windows(input_folder)[:19]
    (num_windows, height, width) = input_windows.shape
    enc = encoder.Encoder(height * width)
    errs = enc.train(input_windows)

    problems = get_problems(input_folder)[:2]
    for i in range(len(problems)):
        out_file = output_folder + '/' + str(i) + '.txt'
        write_problem(problems[i], out_file)
        with open(out_file, 'a') as f:
            f.write('== SDRs ==\n')
            for win in np.concatenate((problems[i]['Input'], problems[i]['Output'])):
                encoded = enc.encode(win.reshape((1, height, width, 1)))
                print(encoded.shape)
                m = np.mean(encoded)
                #print(np.min(encoded), np.mean(encoded), np.max(encoded))
                print(np.sum(encoded))
                encoded = (encoded >= m).astype(int)
                print(np.sum(encoded))
                print(encoded.shape)
                decoded = enc.decode(encoded)


                size = int(len(encoded[0]) ** 0.5)
                for line in encoded.reshape((size, size)):
                    f.write(''.join([str(x) for x in line]) + '\n')
                f.write('\n')
                
                m = np.mean(decoded)
                decoded = (decoded >= m).astype(int)
                print(np.sum(win == decoded.reshape(win.shape)))
                for line in decoded.reshape((height, width)):
                    f.write(''.join([str(x) for x in line]) + '\n')
                f.write('\n')
                f.write('\n')

                print("---------------")



def get_problems(folder_name):
	problems = []
	for i in range(len(os.listdir(folder_name))):
		file_name = '%d.txt' % i
		problem = {}
		with open(os.path.join(folder_name, file_name)) as f:
			lines = f.read().split('\n')
			key = re.match('== (.*) ==', lines[0]).group(1)
			problem[key] = {}
			index = 1
			while lines[index] != '':
				m = re.match('(.*)=(.*)', lines[index])
				problem[key][m.group(1)] = m.group(2).strip()
				index += 1

			problem['Attributes']['result'] = int(problem['Attributes']['result'])
			m = re.match('\((.*), (.*)\)', problem['Attributes']['window_size'])
			problem['Attributes']['window_size'] = (int(m.group(1)), int(m.group(1)))

			while index < len(lines) - 1:
				index += 1
				key = re.match('== (.*) ==', lines[index]).group(1)
				problem[key] = []
				index += 1
				while index < len(lines) - 1:
					new_window = []
					for _ in range(problem['Attributes']['window_size'][0]):
						new_window += [[int(x) for x in lines[index]]]
						index += 1

					problem[key].append(np.array(new_window))
					
					if index < len(lines)-1 and re.match('== (.*) ==', lines[index + 1]):
						break
					index += 1
				problem[key] = np.array(problem[key])
		problems += [problem]
	return problems


def get_windows(folder_name):
	problems = get_problems(folder_name)

	windows = np.append(problems[0]['Input'],
						problems[0]['Output'], 
						axis=0)
	for problem in problems[1:]:
		windows = np.append(windows, problem['Input'], axis=0)
		windows = np.append(windows, problem['Output'], axis=0)
	
	return windows


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--parse")
    parser.add_argument("--sdrs")
    args = parser.parse_args()
	
    problem_dir = '../Problem_images/resized'
	problem_txt_dir = '../Problems_txt'
	output_dir = '../Problems'

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

	