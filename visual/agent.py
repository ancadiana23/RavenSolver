import os
import png


def main():
	problem_directory = '../Problem_images'
	for folder_name in os.listdir(problem_directory):
		folder_name = problem_directory + os.path.sep + folder_name
		
		
		
		for file_name in os.listdir(folder_name):
			print file_name
			#file_name = folder_name + os.path.sep + file_name
			file_name = '../Problem_images/2x1 Basic Problems/2x1BasicProblem01.PNG'
			f = open(file_name)
			r=png.Reader(file=f)
			img_struct = r.read()

			img = list(img_struct[2])
			img = [[2 if l[i + 2] > l[i + 1] and l[i + 2] > l[i] else 
					1 - int(round((l[i] + l[i + 1] + l[i + 2]) / (3 * 255.0)))
						for i in range(0, len(l), 3)] 
						for l in img]
			
			#for l in img[100:150]:
			#	print l[150:200]
			size = (len(img), len(img[0]))
			print size
			x = 0
			y = 0
			while img[x][y] == 0 or \
				  img[x][y] == 1:
				y += 1
				if y== len(img[0]):
					y = 0
					x += 1
			while img[x][y] == 2:
				x += 1
				y += 1
			
			window_size = (x, y)
			print window_size

			while img[x][window_size[1]] != 2:
				x += 1
			while img[window_size[0]][y] != 2:
				y += 1
			window_size = (x - window_size[0], y - window_size[1])
			print window_size

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
			print len(windows)
			
			break
		break
	f.close()

if __name__ == "__main__":
	main()
