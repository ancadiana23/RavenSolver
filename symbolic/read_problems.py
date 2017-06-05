import os
import re


def unique_attributes(problem, problem_attributes):
	if type(problem) is str:
		print problem
	
	for key in problem:
		if key not in problem_attributes:
			if type(problem[key]) is dict:
				problem_attributes[key] = {}
			else:
				problem_attributes[key] = set([])
		if type(problem_attributes[key]) is dict:
			unique_attributes(problem[key], problem_attributes[key])
		else:
			problem_attributes[key].add(problem[key])


def read_content(lines, index, level):
	content = {}
	while index < len(lines):
		line = lines[index].rstrip()
		new_level = 0
		while line[0] == '\t':
			line = line[1:]
			new_level += 1
		if new_level == level:
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
	problem_attributes = {}
	for problem in problems:
		for key in problem['content']:
			unique_attributes(problem['content'][key], problem_attributes)
	
	return problem_attributes

def get_problems():
	problem_dir = "../Problems_txt"
	problems = []
	
	for file in os.listdir(problem_dir):
		new_file = os.path.join(problem_dir, file)
		with open(new_file) as f:
			new_problem = {}
			new_problem['title' ] = f.readline().rstrip()
			new_problem['type'  ] = f.readline().rstrip()
			new_problem['result'] = int(f.readline().rstrip())
			lines = f.readlines()
			new_problem['content'], index = read_content(lines, 0, 0)
			problems.append(new_problem)
	
	return problems, get_unique_attributes(problems)

if __name__ == "__main__":
	main()
