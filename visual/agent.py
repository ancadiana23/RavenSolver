import os
import png

from nupic.research.TP import TP
from nupic.research.temporal_memory import TemporalMemory as TM


def read_problem(problem, png_file, txt_file):
	with open(txt_file) as f:

		problem['title' ] = f.readline().rstrip()
		problem['type'  ] = f.readline().rstrip()
		problem['result'] = int(f.readline().rstrip())

	f = open(png_file)
	r=png.Reader(file=f)
	img_struct = r.read()
	img = list(img_struct[2])
	f.close()

	if img_struct[3]['planes'] == 1:
		palette = [2 if p[2] > p[1] and p[2] > p[0] 
					else (1 - sum(p) / (3 * 255.0)) 
				   for p in img_struct[3]['palette']]
		img = [[palette[x] for x in l] for l in img]
	else:
		img = [[2 if l[i + 2] > l[i + 1] and l[i + 2] > l[i] else 
				1 - int(round((l[i] + l[i + 1] + l[i + 2]) / (3 * 255.0)))
					for i in range(0, len(l), 3)] 
				for l in img]
	
	size = (len(img), len(img[0]))
	x = 0
	y = 0
	while img[x][y] != 2:
		y += 1
		if y== len(img[0]):
			y = 0
			x += 1
	while img[x][y] == 2:
		x += 1
		y += 1
	
	window_size = (x, y)
	while img[x][window_size[1]] != 2:
		x += 1
	while img[window_size[0]][y] != 2:
		y += 1
	window_size = (x - window_size[0], y - window_size[1])
	
	x = 1
	y = 1
	windows = []
	while x < size[0] and y < size[1]:
		if img[x][y] != 2 and \
		   img[x - 1][y    ] == 2 and \
		   img[x - 1][y - 1] == 2 and \
		   img[x    ][y - 1] == 2 and \
		   img[x + 1][y - 1] == 2 and \
		   img[x - 1][y + 1] == 2: 
		    new_window = [img[x + i][y: y + window_size[1]] for i in range(window_size[0])]
		    windows.append(new_window)
		    y += window_size[1]
		    if y >= size[1]:
		    	y = 0
		    	x += window_size[0]
		y += 1
		if y >= size[1]:
			y = 0
			x += 1
	problem['windows'] = windows


def solve(problem):
	pass


def main():
	problem_dir = '../Problem_images'
	problem_txt_dir = '../Problems'
	problems = []
	for folder in os.listdir(problem_dir):
		folder_name = problem_dir + os.path.sep + folder

		for file_name in os.listdir(folder_name):
			problem = {}

			txt_file = os.path.join(problem_txt_dir, folder, 
								file_name.split('.')[0] + '.txt')
			png_file = os.path.join(folder_name, file_name)
			read_problem(problem, png_file, txt_file)
			problems.append(problem)
			#solve(problem)

if __name__ == "__main__":
	main()
