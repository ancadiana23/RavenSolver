import math


def dist(p1, p2):
	a = pow(p1[0] - p2[0], 2)
	b = pow(p1[1] - p2[1], 2)
	return pow(a + b, 0.5)


def circle(fig, x):
	size = len(fig)
	center = math.ceil(len(fig) / 2) - 1
	for i in range(size):
		for j in range(size):
			d = dist((i, j), (center, center))
			if d >= (x - 1) and d <= (x + 1):
				fig[i][j] = 1
			if d < (x - 1):
				fig[i][j] = 2
	return fig


def square(fig, x):
	size = len(fig)
	for i in [x, x + 1, size - x - 1, size - x - 2]:
		for j in range(x, size - x):
			fig[i][j] = 1
	for i in range(x + 2, size - x - 2):
		for j in [x, x + 1, size - x - 1, size - x - 2]:
			fig[i][j] = 1
	for i in range(x + 2, size - x - 2):
		for j in range(x + 2, size - x - 2):
			fig[i][j] = 2
	return fig


def rectangle(fig, x):
	size = len(fig)
	y = 3 * x
	for i in [x, x + 1, size - x - 1, size - x - 2]:
		for j in range(y, size - y):
			fig[i][j] = 1
	for i in range(x, size - x):
		for j in [y, y + 1, size - y - 1, size - y - 2]:
			fig[i][j] = 1
	for i in range(x + 2, size - x - 2):
		for j in range(y + 2, size - y - 2):
			fig[i][j] = 2
	return fig


def triangle(fig, x):
	size = len(fig)
	center = math.ceil(size / 2) - 2
	y = int((size  - (center - x - 1 + 3)) / 2) + 1

	for j in range(x, size - x):
		fig[size - y - 1][j] = 1
	for j in range(x - 1, size - x + 1):
		fig[size - y][j] = 1

	j = x + 1
	i = size - y - 2
	while j != int(size / 2):
		fig[i][j] = 1
		fig[i][j + 1] = 1

		fig[i][size - j - 1] = 1
		fig[i][size - j - 2] = 1
		j += 1
		i -= 1
	if size % 2 == 1:
		fig[i][j] = 1


	start = x + 3
	end = size - x - 3
	i = size - y - 2
	while start < end:
		for j in range(start, end):
			fig[i][j] = 2
		start += 1
		end -= 1
		i -= 1
	return fig


def write_fig(fig, text_file, name):
	string = ""
	for line in fig:
		for x in line:
			string += str(x)
		string += "\n"
	string += "\n\n"
	text_file.write(name + '\n')
	text_file.write(string)


def fill(fig, x1, y1, x2, y2):
	for i in range(x1, x2):
		for j in range(y1, y2):
			fig[i][j] = 1
	return fig


def main(size, file_name):
	folder = "constants/"
	text_file = open(folder + "constants_" + file_name, "w+")
	text_file.write(str(size) + '\n\n')
	for (name, func, x) in [('circle', circle, math.ceil(size / 2.5)), ('square', square, math.ceil(size / 7)), \
							('rectangle', rectangle, math.ceil(size / 13)), ('triangle', triangle, math.ceil(size / 20))]:
		fig = [[0 for _ in range(size)] for _ in range(size)]
		fig = func(fig, x)
		write_fig(fig, text_file, name)
	text_file.close()

	text_file = open(folder + "fills_" + file_name, "w+")
	text_file.write(str(size) + '\n\n')
	half = math.ceil(size / 2) - 1
	for (name, x1, y1, x2, y2) in [('none', 0, 0, 0, 0), ('full', 0, 0, size, size), \
								   ('horizontal_half_1', 0, 0, half, size), ('horizontal_half_2', half, 0, size, size), \
								   ('vertical_half_1', 0, 0, size, half), ('vertical_half_2', 0, half, size, size)]:
		fig = [[0 for _ in range(size)] for _ in range(size)]
		fig = fill(fig, x1, y1, x2, y2)
		write_fig(fig, text_file, name)
	text_file.close()


if __name__ == "__main__":
	for (size, file_name) in [(75, 'L.txt'), (50, 'M.txt'), (25, 'S.txt')]:
		main(size, file_name)

