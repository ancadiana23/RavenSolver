from argparse import ArgumentParser
from nupic.algorithms.backtracking_tm import BacktrackingTM as BTM
from random import randint

import read_problems
import numpy as np

def create_SDR(window, problem_attributes, SDR, match=True):
	"""
	Create the SDR that encodes a window.

	Args:
	window: dictionary describing a window
	problem_attributes: dictionary that encodes all the possibile attributes in all the windows
	SDR: list that will contain the result
	match: boolean that states of the current branch in the 
		problem_attributes branch was previosly mqatched in the window;
		used to add 0s for all other possbile attributes

	Returns:
	a list of 0s and 1s that encodes the input window
	"""

	if match:
		# if the previous attributes on the current branched are present in the window
		if type(problem_attributes) is dict:
			for key in sorted(problem_attributes.keys()):
				# for every param on the current level of the current branch
				# check if it is in the window
				if key in window:
					# if it is, continue down its branch
					create_SDR(window[key], problem_attributes[key], SDR, match)
				else:
					# if it is not, add 0s for all the nodes on its branch 
					create_SDR({}, problem_attributes[key], SDR, False)
		else:
			# if a leaf was reached add 1 for the attribute on that leaf 
			# and 0 for all the other possibilities
			attributes = sorted(list(problem_attributes))
			SDR.extend(int(x == window) for x in attributes)
	else:
		# if the current branch was not previously matched then just add 0s 
		# in order to preserve a consistent structure for all SDRs
		if type(problem_attributes) is dict:
			for key in problem_attributes:
				create_SDR({}, problem_attributes[key], SDR, match)
		else:
			SDR.extend([0] * len(problem_attributes))


def run_solver_BKTM(problems, param):
	"""
	Use the Backtracking Temporal Memory algorithm to solve the problems.
	
	Args:
	problems: list of problems to be solved
	param: tuple of parameters for the algorithm

	Returns:
	number of correctly solved problems
	"""

	#print(param)
	m = len(problems)
	SDRlen = len(problems[0]['SDRs']['input'][0])
	results = []
	matches = 0
	i = 0

	# initialize the algorithm
	tm = BTM(numberOfCols=SDRlen, cellsPerColumn=param[0],
				initialPerm=param[1], connectedPerm=param[2],
				minThreshold=param[3], newSynapseCount=param[4],
				permanenceInc=param[5], permanenceDec=param[6],
				activationThreshold=param[7])

	for problem in problems:
		'''
		print('')	
		
		print(problem['title'])
		print(np.sum(problem['SDRs']['input'], axis = 1))
		print(np.sum(problem['SDRs']['output'], axis = 1))
		'''
		
		# run the input winodws through the algorithm
		tm.compute(problem['SDRs']['input'][0], enableLearn=True, enableInference=True)
		tm.compute(problem['SDRs']['input'][1], enableLearn=True, enableInference=True)
		tm.compute(problem['SDRs']['input'][2], enableLearn=True, enableInference=True)
		
		# get the predicted state (a list of length SDRlen containing numbers between 0 and 1)
		predictedCells = tm.predict(1)
		
		# compute the match between every output window and the predicted state
		match = [np.sum(x * predictedCells) for x in problem['SDRs']['output']]
		#print match
		
		max_match = max(match)

		vote = np.argmax(max_match)
		#if np.sum(predictedCells):
		#	print(problem['title'],problem['result'], vote, np.sum(predictedCells), sum(problem['SDRs']['output'][problem['result'] - 1]))

		results.append(vote)
		tm.compute(problem['SDRs']['output'][vote], enableLearn=True, enableInference=False)
		i += 1

	return sum([results[i] == (problems[i]['result'] - 1) for i in range(m)])


def run_solver_TM(problems, param):
	"""
	Use the Temporal Memory algorithm to solve the problems.
	
	Args:
	problems: list of problems to be solved
	param: tuple of parameters for the algorithm

	Returns:
	number of correctly solved problems
	"""

	m = len(problems)
	SDRlen = len(problems[0]['SDRs']['input'][0])
	results = []
	if len(param) != 8:
		print(param)

	# initialize algorithm
	tm = TM(numberOfCols=SDRlen, cellsPerColumn=param[0],
				initialPerm=param[1], connectedPerm=param[2],
				minThreshold=param[3], newSynapseCount=param[4],
				permanenceInc=param[5], permanenceDec=param[6],
				activationThreshold=param[7],
				globalDecay=0, burnIn=1,
				checkSynapseConsistency=False,
				pamLength=10)

	matches = 0
	for problem in problems:
		# run the input winodws through the algorithm
		tm.compute(problem['SDRs']['input'][0], learn = True)
		tm.compute(problem['SDRs']['input'][1], learn = True)
		tm.compute(problem['SDRs']['input'][2], learn = True)
		
		# get prediction cells and create a state containing one for every 
		# predicted cell and 0 otherwise
		# compare the state with every output window and choose the best match
		predictedCells = tm.getPredictiveCells()
		if predictedCells:
			predict = [0] * SDRlen
			for x in predictedCells:
				predict[x] = 1
			match = [np.sum(np.logical_and(x, predict)) for x in problem['SDRs']['output']]
		
			vote = match.index(max(match)) + 1
		else:
			vote = 0

		# run the correct result throught the algorithm
		tm.compute(problem['SDRs']['output'][problem['result'] - 1], learn = True)

		# append the best match to the result list
		results.append(vote)

	# return the bumber of correctly solved problems
	return sum([results[i] == problems[i]['result'] for i in range(m)])


def local_search(problems, algorithm, initial_param):
	"""
	Search or the optimal parameters using the algorithm 'Simulated annealing'.

	Args:
	problems: the list of problems to be solved
	algorithm: the function that calls the learning algorithm that is tested
	initial_param: a tuple containing a initial parameters for the algorthm

	Returns:
	best_result: the result of the algorithms using the best parameters
	best_param: a tuple containing the best parameters for the algorithm
	"""

	best_result = algorithm(problems, initial_param)
	best_param = initial_param
	
	# intervals and steps for every parameter
	steps = [1, 0.1, 0.1, 1, 1, 0.1, 0.1, 1]
	limits = [(1, 64), (0.1, 1), (0.1, 1), (1, 64), (1, 64), (0.1, 1), (0.1, 1), (1, 64)]

	probability = 4.0
	effort = 100 	# number of iterations
	while effort != 0:
		effort -= 1
		for i in range(len(best_param)):
			# for every parameter change it by one step in every direction and test the algorithm
			# if the result is better than the previous one then keep the change
			# otherwise keep the change with a certain probability
			# decrease the probability every time a worse result is selected
			for sign in [1, -1]:
				x = best_param[i] + sign * steps[i]
				if  x >= limits[i][0] and x <= limits[i][1]:
					new_param = best_param[:i] + ((best_param[i] + sign * steps[i]),) + best_param[i + 1:]

					result = algorithm(problems, new_param)
					if result > best_result:
						best_param = new_param
						best_result = result
					else:
						r = randint(0, 100)
						if r < 100 / probability:
							probability += 0.5
							best_param = new_param
							best_result = result
	return best_result, best_param


def find_optimal_param(problems, algorithm):
	"""
	Iterate to various cominations of values for the parameters in other to 
	choose the optimal one.

	Args:
	problems: list of dictionaries describing the problems to be solved

	"""

	SDRlen = len(problems[0]['SDRs']['input'][0])
	m = len(problems)
	params = [(cellsPerColumn, initialPerm / 10.0, connectedPerm / 10.0, 
				minThreshold, newSynapseCount, 
				permanenceInc / 10.0, permanenceDec /10.0,
				activationThreshold)
				for activationThreshold in range(1, 16, 8)
				for minThreshold in range(2, activationThreshold, 8)
				for cellsPerColumn in range(2, 16, 8)
				for initialPerm in range(1, 9, 4)
				for connectedPerm in range(1, 9, 4)
				for newSynapseCount in range(2, 16, 8)
				for permanenceInc in range(1, 9, 4)
				for permanenceDec in range(1, 9, 4)]

	print(len(params))

	matches_list = []
	max_matches = 0
	best_param = ()
	index = 0.0
	
	for param in params:
		matches = algorithm(problems, param)

		index += 1
		matches_list.append(matches)
		if matches > max_matches:
			max_matches = matches
			best_param = param
	
	f = open("output.csv", "w+")
	for i in range(len(matches_list)):
		s = ",".join([str(x) for x in params[i]])
		s += ',' + str(matches_list[i]) + '\n'
		f.write(s)
	return best_param


def run(args):
	"""
	Run the agent:
		-> read the problems
		-> create SDRs
		-> choose parameters
		-> apply the algorithm to the problems
	"""

	# choose the learning algorithm
	if args.algorithm == 'BKTM':
		algorithm = run_solver_BKTM
	elif args.algorithm == 'TM':
		algorithm = run_solver_TM

	# read the problems and create SDRs vor every window in every problem
	problems, problem_attributes = read_problems.get_problems()
	SDRs = []	
	for problem in problems:
		problem['SDRs'] = {'input': [],
						   'output': []}
		
		for (x, y) in [('A', 'input'), ('1', 'output')]:
			# for every input window, desbribed by letter A-C,
			# and for every output window, described by numbers 1-6,
			# create an SDR
			index = ord(x)
			while chr(index) in problem['content']:
				new_SDR = []
				create_SDR(problem['content'][chr(index)], problem_attributes, new_SDR)
				problem['SDRs'][y].append(np.array(new_SDR))
				index += 1
	
	m = len(problems)
	# choose the parameters for the learning algorithm
	#param = find_optimal_param(problems, algorithm)
	param = (2, 0.5, 0.1, 2, 10, 0.5, 0.1, 9)
	
	# run the algorithm
	result = algorithm(problems, param)
	
	print(result, param)
	result, param = local_search(problems, algorithm, param)
	print(result, param)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--algorithm", type = str, default = 'BKTM')
	args = parser.parse_args()

	run(args)
