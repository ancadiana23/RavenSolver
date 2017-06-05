import numpy as np
from argparse import ArgumentParser
from nupic.research.TP import TP
from nupic.research.temporal_memory import TemporalMemory as TM
from nupic.research.spatial_pooler import SpatialPooler
from BacktrackingTM import BacktrackingTM
from parse_input import get_problems


class SingleExperiments:
	def print_window(window):
		for line in window:
			print(''.join([str(x) for x in line]))
		print('\n')

	def experiment4(self):
		print("Running experiment 4")
		
		numColls = 16
		x = np.zeros(numColls).astype(int)
		y = np.zeros(numColls).astype(int)
		x[:numColls/2] = 1
		y[numColls/2:] = 1
		print(x)
		print(y)
			
		tm = BacktrackingTM(numberOfCols=numColls, cellsPerColumn=2,
					initialPerm=0.5, connectedPerm=0.5,
					minThreshold=10, newSynapseCount=10,
					permanenceInc=0.1, permanenceDec=0.0,
					activationThreshold=8,
					globalDecay=0, burnIn=1,
					checkSynapseConsistency=False,
					pamLength=numColls)
		for i in range(5):
			tm.compute(x, enableLearn=True, computeInfOutput=True)
			predictedCells1 = tm.getPredictedState()
			#tm.compute(x, learn=True)
			#predictedCells1 = tm.getPredictiveCells()

			if np.sum(predictedCells1) > 0:
				print(predictedCells1)
				#print("!!!!", np.sum(predictedCells1))

			tm.compute(y, enableLearn=True, computeInfOutput=True)
			predictedCells2 = tm.getPredictedState()
			#tm.compute(x, learn=True)
			#predictedCells2 = tm.getPredictiveCells()
			if np.sum(predictedCells2):
				print(predictedCells2)
				#print("??", np.sum(predictedCells2))		


	def experiment3(self):
		print("Running experiment 3")

		numColls = 16
		s = numColls ** (0.5)
		print(s)

		'''
		x = np.zeros((s, s)).astype(int)
		x[ 2, 2:-2] = 1
		x[-3, 2:-2] = 1
		x[2:-2,  2] = 1
		x[2:-2, -3] = 1
		print(x)
		x = x.reshape(numColls)
		
		y = np.zeros((s, s)).astype(int)
		y[2:-2, 2:-2] = 1
		print(y)
		y = y.reshape(numColls)
		'''
		x = np.zeros(numColls).astype(int)
		y = np.zeros(numColls).astype(int)
		x[:numColls/2] = 1
		y[numColls/2:] = 1
		
		print(x)
		print(y)

		tm = BacktrackingTM(numberOfCols=numColls, cellsPerColumn=2,
						initialPerm=0.5, connectedPerm=0.5,
						minThreshold=10, newSynapseCount=10,
						permanenceInc=0.1, permanenceDec=0.0,
						activationThreshold=8,
						globalDecay=0, burnIn=1,
						checkSynapseConsistency=False,
						pamLength=10)
		
		for i in range(10):
			print("---- 1 ------")
			tm.compute(x, enableLearn = True, computeInfOutput = True)
			predictedCells1 = tm.getPredictedState()
			print(sum(predictedCells1))
			#res1 = np.transpose(predictedCells1)

			#res1 = res1.reshape((len(res1), s, s))
			#print_window(res1[0])
			#print_window(res1[1])
			print('\n')

			print("---- 2 ------")
			tm.compute(y, enableLearn = True, computeInfOutput = True)
			predictedCells2 = tm.getPredictedState()
			print(sum(predictedCells2))
			#res2 = np.transpose(predictedCells2)
			
			#res2 = res2.reshape((len(res2), s, s))
			#print_window(res2[0])
			#print_window(res2[1])
			print('\n')


	def experiment2(self):
		print("Running experiment 2")

		problem1 = np.zeros((32, 32)).astype(int)	

		problem1[10, 10:-10] = 1
		problem1[-10, 10:-10] = 1
		problem1[10:-10, 10] = 1
		problem1[10:-10, -10] = 1
		problem1 = problem1.reshape(32 * 32)

		problem2 = np.zeros((32, 32)).astype(int)		
		problem2[10:-10, 10:-10] = 1
		problem2 = problem2.reshape(32 * 32)
		print(problem1)
		print(problem2)
		tm = TM()
		for _ in range(100):
			tm.compute(problem1, learn = True)
			predictiveCells = tm.getPredictiveCells()
			if predictiveCells != []:
				print("!!!!!")
				print(predictiveCells)
			tm.compute(problem2, learn = True)


	def experiment1(self):
		print("Running experiment 1")

		print "Getting problems...",
		folder_name = '../Problems'
		problems = []
		get_problems(folder_name, problems)
		print("Finished")
		
		inputDim = problems[0]['Input'][0].shape
		sp = SpatialPooler(inputDimensions = inputDim, columnDimensions = (inputDim[0] / 8, ))
		print(sp.getColumnDimensions())
	
		for problem in problems[:1]:
			for _ in range(1):
				windows = problem['Input'] + problem['Output']
				print(len(windows))
				for window in problem['Input'] + problem['Output']:
					output = np.zeros(sp.getColumnDimensions(), dtype="int")
					sp.compute(window, learn = True, activeArray = output)
					if sum(output) != 0:
						print(sum(output))


	def run_solver_TP(problems, param):
		print("Running TP solver")

		m = len(problems)
		results = []
		matches = 0
		
		SDRlen = problems[0]['Attributes']['window_size'][0] * problems[0]['Attributes']['window_size'][1]
		tp = TM(numberOfCols=SDRlen, cellsPerColumn=param[0],
				initialPerm=param[1], connectedPerm=param[2],
				minThreshold=param[3], newSynapseCount=param[4],
				permanenceInc=param[5], permanenceDec=param[6],
				activationThreshold=param[7],
				globalDecay=0, burnIn=1,
				checkSynapseConsistency=False,
				pamLength=10)
		print param
		for problem in problems:
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
			
		return sum([results[i] == problems[i]['Attributes']['result'] for i in range(m)])

		return 0


	def find_optimal_param(problems, algorithm):
		print("Finding optimal params")

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


	def main(self, args):
		if args.find_optimal_param:
			algorithm = self.run_solver_TP
			folder_name = '../Problems'
			problems = []
			get_problems(folder_name, problems)

			param = find_optimal_param(problems, algorithm)
			run_solver_TP(problems, param)

		
		experiments = [self.experiment1, self.experiment2, 
						self.experiment3, self.experiment4]
		
		if args.experiment != 0:
			experiments[args.experiment - 1]()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--find_optimal_param", action = "store_true")
    parser.add_argument("--experiment", type = int, default = 0, help="Experiment number")
    args = parser.parse_args()

    se = SingleExperiments()
    se.main(args)
	
