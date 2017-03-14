import read_problems
import numpy as np

from nupic.research.TP import TP
from nupic.research.temporal_memory import TemporalMemory as TM
def create_SDR(problem, problem_attributes, SDR, match=True):
	if match:
		if type(problem_attributes) is dict:
			for key in sorted(problem_attributes.keys()):
				if key in problem:
					create_SDR(problem[key], problem_attributes[key], SDR, match)
				else:
					create_SDR({}, problem_attributes[key], SDR, False)
		else:
			attributes = sorted(list(problem_attributes))
			SDR.extend(int(x == problem) for x in attributes)
	else:
		if type(problem_attributes) is dict:
			for key in problem_attributes:
				create_SDR({}, problem_attributes[key], SDR, match)
		else:
			SDR.extend([0] * len(problem_attributes))

def run_solver_TP(problems, param):
	SDRlen = len(problems[0]['SDRs']['input'][0])
	results = []
	tp = TP(numberOfCols=SDRlen, cellsPerColumn=param[0],
				initialPerm=param[1], connectedPerm=param[2],
				minThreshold=param[3], newSynapseCount=param[4],
				permanenceInc=param[5], permanenceDec=param[6],
				activationThreshold=param[7],
				globalDecay=0, burnIn=1,
				checkSynapseConsistency=False,
				pamLength=10)

	matches = 0
	for problem in problems:
		tp.compute(problem['SDRs']['input'][0], enableLearn = True, computeInfOutput = False)
		tp.compute(problem['SDRs']['input'][1], enableLearn = True, computeInfOutput = False)

		tp.compute(problem['SDRs']['input'][2], enableLearn = False, computeInfOutput = True)
		
		predictedCells = tp.getPredictedState()
		predictedCells = [sum(x) for x in predictedCells]
		
		match = [np.sum(np.logical_and(x, predictedCells)) for x in problem['SDRs']['output']]
		
		vote = match.index(max(match))
		results.append(vote)

	return results

def run_solver_TM(problems, param):
	SDRlen = len(problems[0]['SDRs']['input'][0])
	results = []
	if len(param) != 8:
		print param
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
		tm.compute(problem['SDRs']['input'][0], learn = True)
		tm.compute(problem['SDRs']['input'][1], learn = True)

		tm.compute(problem['SDRs']['input'][2], learn = False)
		
		predictedCells = tm.getPredictiveCells()
		if predictedCells:
			predict = [0] * SDRlen
			for x in predictedCells:
				predict[x] = 1
			match = [np.sum(np.logical_and(x, predict)) for x in problem['SDRs']['output']]
		
			vote = match.index(max(match))
		else:
			vote = -1
		results.append(vote)

	return results


def find_optimal_param(problems, algorithm):
	SDRlen = len(problems[0]['SDRs']['input'][0])
	m = len(problems)
	params = [(cellsPerColumn, initialPerm / 10.0, connectedPerm / 10.0, 
				minThreshold, newSynapseCount, 
				permanenceInc / 10.0, permanenceDec /10.0,
				activationThreshold)
				for activationThreshold in range(1, 9, 3)
				for minThreshold in range(2, 9, 3)
				for cellsPerColumn in range(2, 16, 4)
				for initialPerm in range(1, 9, 4)
				for connectedPerm in range(1, 9, 4)
				for newSynapseCount in range(2, 16, 8)
				for permanenceInc in range(1, 9, 4)
				for permanenceDec in range(1, 9, 4)]

	print len(params)

	matches_list = []
	max_matches = 0
	best_param = ()
	index = 0.0
	
	for param in params:
		results = algorithm(problems, param)
		matches = sum([results[i] == problems[i]['result'] for i in range(m)])

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


def run():
	algorithm = run_solver_TP
	problems, problem_attributes = read_problems.get_problems()
	SDRs = []
	for problem in problems:
		problem['SDRs'] = {'input': [],
						   'output': []}
		
		for (x, y) in [('A', 'input'), ('1', 'output')]:
			index = ord(x)
			while chr(index) in problem['content']:
				new_SDR = []
				create_SDR(problem['content'][chr(index)], problem_attributes, new_SDR)
				problem['SDRs'][y].append(np.array(new_SDR))
				index += 1
	m = len(problems)
	param = find_optimal_param(problems, algorithm)
	print param
	results = algorithm(problems, param)
	matches = sum([results[i] == problems[i]['result'] for i in range(m)])
	print(matches)


if __name__ == "__main__":
	run()
