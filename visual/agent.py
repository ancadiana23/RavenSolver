import os
import png
import re

import numpy as np

from nupic.research.TP import TP
from nupic.research.temporal_memory import TemporalMemory as TM

from BacktrackingTM import BacktrackingTM

def read_problems(folder_name, problems):
	for file_name in os.listdir(folder_name):
		problem = {}
		with open(os.path.join(folder_name, file_name)) as f:
			lines = f.read().split('\n')
			key = re.match('== (.*) ==', lines[0]).group(1)
			problem[key] = {}
			index = 1
			while lines[index] != '':
				m = re.match('(.*)=(.*)', lines[index])
				#print "===", lines[index], '==='
				problem[key][m.group(1)] = m.group(2)
				index += 1

			problem['Attributes']['result'] = int(problem['Attributes']['result'])
			m = re.match('\((.*), (.*)\)', problem['Attributes']['window_size'])
			problem['Attributes']['window_size'] = (int(m.group(1)), int(m.group(1)))

			#print problem

			while index < len(lines) - 1:
				index += 1
				#print lines[index - 1]
				#print lines[index]
				key = re.match('== (.*) ==', lines[index]).group(1)
				problem[key] = []
				index += 1
				while index < len(lines) - 1:
					new_window = []
					#print index, lines[index]
					for _ in range(problem['Attributes']['window_size'][0]):
						new_window += [int(x) for x in lines[index]]
						index += 1

					if index < len(lines)-1 and re.match('== (.*) ==', lines[index + 1]):
						break
					#print index, lines[index]
					index += 1
					problem[key].append(np.array(new_window))
					#index += 1
	
		#print problem
		problems += [problem]



def run_solver_TP(problems, param):
	m = len(problems)
	results = []
	matches = 0
	
	SDRlen = problems[0]['Attributes']['window_size'][0] * problems[0]['Attributes']['window_size'][1]
	tp = TP(numberOfCols=SDRlen, cellsPerColumn=param[0],
			initialPerm=param[1], connectedPerm=param[2],
			minThreshold=param[3], newSynapseCount=param[4],
			permanenceInc=param[5], permanenceDec=param[6],
			activationThreshold=param[7],
			globalDecay=0, burnIn=1,
			checkSynapseConsistency=False,
			pamLength=10)
	print param
	for problem in problems[:1]:
		predictiveCells = [[]]
		aux = 0
		while np.sum(predictiveCells) < 5 and aux < 1000:
			aux += 1
			for window in problem['Input']:
				tp.compute(window, enableLearn = True, computeInfOutput = True)
			predictiveCells = tp.getPredictedState()
			tp.compute(problem['Output'][problem['Attributes']['result'] - 1], enableLearn = True, computeInfOutput = False)
			tp.reset()
		print aux

		if np.sum(predictiveCells) > 0:
			print np.sum(predictiveCells)
		'''
		for window in problem['Input']:
			tp.compute(window, enableLearn=False, computeInfOutput = True)

		predictiveCells = tp.getPredictedState()
		#print len(predictiveCells)
		if len(predictiveCells) > 0:
			#predictedCells = [sum(x) for x in predictedCells]
			
			predictedCells = [0] * SDRlen
			for cell in predictiveCells:
				predictedCells[cell] = 1
			
			match = [np.sum(np.logical_and(window, predictedCells)) for window in problem['Output']]
			#print match
			
			max_matche = max(match)
			vote = match.index(max_matche) + 1
		else:
			vote = -1

		print vote, problem['Attributes']['result']
		
		results.append(vote)
		tp.compute(problem['Output'][problem['Attributes']['result'] - 1], learn=False)
		break
		
	return sum([results[i] == problems[i]['Attributes']['result'] for i in range(1)])
	#return sum([results[i] == problems[i]['Attributes']['result'] for i in range(m)])
	'''
	return 0


def find_optimal_param(problems, algorithm):
	SDRlen = problems[0]['Attributes']['window_size'][0] * problems[0]['Attributes']['window_size'][1]
	m = len(problems)
	params = [(cellsPerColumn, initialPerm / 10.0, connectedPerm / 10.0, 
				minThreshold, newSynapseCount, 
				permanenceInc / 10.0, permanenceDec /10.0,
				activationThreshold)
				for activationThreshold in range(1, 16, 8)
				for minThreshold in range(2, activationThreshold, 8)
				for cellsPerColumn in range(8, 64, 8)
				for initialPerm in range(1, 9, 4)
				for connectedPerm in range(1, 9, 4)
				for newSynapseCount in range(2, 16, 8)
				for permanenceInc in range(1, 9, 4)
				for permanenceDec in range(1, 9, 4)]
	print len(params)
	matches_list = []
	max_matches = -1
	best_param = ()
	index = 0.0
	
	for param in params:
		matches = algorithm(problems, param)
		print matches
		index += 1
		matches_list.append(matches)
		if matches > max_matches:
			max_matches = matches
			best_param = param

	return best_param


def main():
	algorithm = run_solver_TP

	folder_name = '../Problems'
	problems = []
	read_problems(folder_name, problems)
	print(problems[0]['Input'][0].shape)
	param = find_optimal_param(problems, algorithm)
	#print run_solver_TP(problems, param)


if __name__ == "__main__":
	main()
