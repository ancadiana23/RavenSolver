def serialize_att(problem_attributes, result, partial_key=''):
	if type(problem_attributes) is dict:
		for key in problem_attributes:
			serialize_att(problem_attributes[key], result, partial_key + key)
	else:
		result[partial_key] = problem_attributes



def run():
		'''

	problem_attributes_s = {}
	serialize_att(problem_attributes, problem_attributes_s)

	serialized_problems = []
	for problem in problems:
		problem_s = {}	
		serialize_att(problem, problem_s)
		for key in problem_attributes:
			if key not in problem_s:
				problem_s[key] = "None"
		serialized_problems = problem_s

	'''
