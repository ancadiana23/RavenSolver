import os

def hex2dec(s):
	return int(s.encode('hex'), 16)			

def read_png_image(file_name, img):
	f = open(file_name)
	content = f.read()
	f.close()
	if [ord(x) for x in content[:8]] != [137, 80, 78, 71, 13, 10, 26, 10]:
		print("Error the file is not PNG")
		return 0

	idx = 8
	while idx < len(content):
		chunck_size = hex2dec(content[idx: idx + 4])
		chunck_type = content[idx + 4: idx + 8]
		chunck_data = content[idx + 8: idx + chunck_size + 8 - 1]
		
		if chunck_type in img:
			img[chunck_type] += chunck_data
		else:
			img[chunck_type] = chunck_data

		if chunck_type == "IHDR":
			img['width']  = hex2dec(chunck_data[0:4])
			img['height'] = hex2dec(chunck_data[4:8])
			
		idx += chunck_size + 12
		if chunck_type == 'IEND':
			break

	print img.keys()
	return 1

def main():
	problem_directory = '../Problem_images'
	for folder_name in os.listdir(problem_directory):
		folder_name = problem_directory + os.path.sep + folder_name
		
		print folder_name
		
		for file_name in os.listdir(folder_name):
			print file_name
			file_name = folder_name + os.path.sep + file_name
			img = {}
			read_png_image(file_name, img)

			break
		break


if __name__ == "__main__":
	main()