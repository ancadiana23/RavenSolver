import numpy as np
from argparse import ArgumentParser
from nupic.algorithms.temporal_memory import TemporalMemory as TM
from nupic.algorithms.backtracking_tm import BacktrackingTM as BTM
from nupic.algorithms.spatial_pooler import SpatialPooler
from parse_images import get_problems


class SingleExperiments:
	def print_window(self, window):
		for line in window:
			print(''.join([str(x) for x in line]))
		print('\n')


	def experiment6(self):
		n = 15
		tm = BTM(numberOfCols=n, cellsPerColumn=3,
                    initialPerm=0.5, connectedPerm=0.5,
                    minThreshold=4, newSynapseCount=10,
                    activationThreshold=5,
                    pamLength=10)
		x = np.zeros(n)
		y = np.zeros(n)
		z = np.zeros(n)

		x[:5] = 1
		y[5:10] = 1
		z[10:] = 1
		print(x)
		print(y)
		print(z)
		for _ in range(10):
			print('-------------')
			tm.compute(x, enableLearn=True, enableInference=True)
			print(x.astype(int))
			print((tm.predict(1) * 100).astype(int))
			print('')
			tm.compute(y, enableLearn=True, enableInference=True)
			print(y.astype(int))
			print((tm.predict(1) * 100).astype(int))
			print('')
			tm.compute(x, enableLearn=True, enableInference=True)
			print(x.astype(int))
			print((tm.predict(1) * 100).astype(int))
			print('')
			tm.compute(z, enableLearn=True, enableInference=True)
			print(z.astype(int))
			print((tm.predict(1) * 100).astype(int))
			print('')


	def experiment5(self):
		n = 8

		tm = BTM(numberOfCols=n * n, cellsPerColumn=3,
                    initialPerm=0.5, connectedPerm=0.5,
                    minThreshold=10, newSynapseCount=10,
                    activationThreshold=10,
                    pamLength=10)
		'''
		tm = BTM(numberOfCols=n * n, cellsPerColumn=4,
                    initialPerm=0.5, connectedPerm=0.5,
                    minThreshold=4, newSynapseCount=5,
                    permanenceInc=0.1, permanenceDec=0.0,
                    activationThreshold=5,
                    globalDecay=0, burnIn=1,
                    checkSynapseConsistency=True)
		'''
		x = np.zeros((n, n))
		y = np.zeros((n, n))

		x[1:-1,  1] = 1
		x[1:-1, -2] = 1
		x[ 1, 1:-1] = 1
		x[-2, 1:-1] = 1

		y[1:-1, 1:-1] = 1
		print(x.astype(int))
		print(y.astype(int))

		x = x.reshape((-1))
		y = y.reshape((-1))

		for _ in range(10):
			tm.compute(x, enableLearn = True, enableInference = True)
			predict = ((tm.predict(1).reshape((n, n))) * 100).astype(int)
			print(predict)
			tm.compute(y, enableLearn = True, enableInference = True)
			predict = ((tm.predict(1).reshape((n, n))) * 100).astype(int)
			print(predict)
			print('\n')

		'''
		x = np.zeros((5, tm.numberOfCols), dtype="uint32")
		for i in range(5):
			x[0,1:9]  = 1   # Input SDR representing "A", corresponding to columns 0-9
			x[1,11:19] = 1   # Input SDR representing "B", corresponding to columns 10-19
			x[2,21:29] = 1   # Input SDR representing "C", corresponding to columns 20-29
			x[3,31:39] = 1   # Input SDR representing "D", corresponding to columns 30-39
			x[4,41:49] = 1
		print(x)
		print('\n')
		'''
		'''
		for i in range(10):
			for j in [2, 3, 4, 1, 0]:
				tm.compute(x[j], enableLearn = True, enableInference = True)
			predict = (tm.predict(5) > 0).astype(int)
			print(predict)
			print('\n')

			#tm.reset()
		'''

	def experiment4(self):
		print("Running experiment 4")

		'''
		s = tuple([6, 6])
		x = np.zeros(s).astype(int)
		y = np.zeros(s).astype(int)

		x[1:-1,  1] = 1
		x[1:-1, -2] = 1
		x[ 1, 1:-1] = 1
		x[-2, 1:-1] = 1

		y[1:-1, 1:-1] = 1
		'''
		numColls = 10
		x = np.zeros(numColls).astype(int)
		y = np.zeros(numColls).astype(int)
		z = np.zeros(numColls).astype(int)
		x[:numColls/2] = 1
		y[numColls/2:] = 1
		#z[2 * numColls / 3:] = 1

		print(x)
		print(y)
		#print(z)

		#x = x.reshape((-1))
		#y = y.reshape((-1))

		tm = BTM(numberOfCols=len(x), cellsPerColumn=3,
					initialPerm=0.5, connectedPerm=0.5,
					minThreshold=10, newSynapseCount=10,
					permanenceInc=0.1, permanenceDec=0.0,
					activationThreshold=8,
					globalDecay=0, burnIn=1,
					checkSynapseConsistency=False,
					pamLength=1)

		for i in range(10):
			tm.compute(x, enableLearn=True, enableInference=True)
			predictedCells1 = tm.getPredictedState()
			print(np.transpose(predictedCells1))
			tm.compute(y, enableLearn=True, enableInference=True)
			#tm.compute(z, enableLearn=True, enableInference=True)



			'''
			predict = (tm.predict(4).reshape((-1, 6, 6)) > 0).astype(int)
			print(predict[0, :, :])
			print(predict[1, :, :])
			print(predict[2, :, :])
			print(predict[3, :, :])
			print('\n')
			'''
			'''
			predict = tm.infer(x).reshape((6, 6, 2))
			print(predict[:, :, 0])
			print(predict[:, :, 1])
			print('\n')
			predict = tm.infer(y).reshape((6, 6, 2))
			print(predict[:, :, 0])
			print(predict[:, :, 1])
			print('\n\n\n')
			'''

			'''
			tm.compute(x, enableLearn=True, enableInference=True)
			predictedCells1 = tm.getPredictedState()
			print(np.transpose(predictedCells1).reshape((2, 6, 6))[1])

			tm.compute(y, enableLearn=True, enableInference=True)
			predictedCells2 = tm.getPredictedState()
			print(np.transpose(predictedCells2).reshape((2, 6, 6))[1])
			#tm.reset()
			'''

	def experiment3(self):
		print("Running experiment 3")

		numColls = 16
		s = int(numColls ** (0.5))
		print(s)

		x = np.zeros(numColls).astype(int)
		y = np.zeros(numColls).astype(int)
		x[:numColls/2] = 1
		y[numColls/2:] = 1

		print(x)
		print(y)

		tm = TM(numberOfCols=numColls, cellsPerColumn=2,
						initialPerm=0.5, connectedPerm=0.5,
						minThreshold=8, newSynapseCount=10,
						permanenceInc=0.1, permanenceDec=0.0,
						activationThreshold=8,
						globalDecay=0, burnIn=1,
						checkSynapseConsistency=False,
						pamLength=10)

		for i in range(100):
			print("---- 1 ------")
			tm.compute(x, learn=True)
			predictedCells1 = tm.getPredictiveCells()
			print(sum(predictedCells1))
			print(np.transpose(predictedCells1))

			print('\n')

			print("---- 2 ------")
			tm.compute(y, learn=True)
			predictedCells2 = tm.getPredictiveCells()
			print(sum(predictedCells2))
			print(np.transpose(predictedCells2))
			print('\n')


	def experiment2(self):
		print("Running experiment 2")

		problem1 = np.zeros((32, 32)).astype(int)

		problem1[10, 10:-10] = 1
		problem1[-10, 10:-10] = 1
		problem1[10:-10, 10] = 1
		problem1[10:-10, -10] = 1
		for line in problem1:
			print(line)
		print('\n')
		problem1 = problem1.reshape(32 * 32)

		problem2 = np.zeros((32, 32)).astype(int)
		problem2[10:-10, 10:-10] = 1
		for line in problem2:
			print(line)
		problem2 = problem2.reshape(32 * 32)

		tm = TM()
		for i in range(100):
			print(i)
			tm.compute(problem1, learn = True)
			predictiveCells = tm.getPredictiveCells()
			if predictiveCells != []:
				print("!!!!!")
				print(predictiveCells)
			tm.compute(problem2, learn = True)


	def experiment1(self):
		print("Running experiment 1")

		print "Getting problems...",
		folder_name = 'Data/Problems'
		problems = get_problems(folder_name)
		print("Finished")

		inputDim = problems[0]['Input'][0].shape
		print(inputDim)
		sp = SpatialPooler(inputDimensions = inputDim, columnDimensions=(32, 32))
		print(sp.getColumnDimensions())

		for problem in problems[:1]:
			windows = np.concatenate((problem['Input'], problem['Output']))
			for _ in range(1):
				for window in windows:
					output = np.zeros(32 * 32, dtype="int")
					print(output.shape)
					sp.compute(window, learn = True, activeArray=output)
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
						self.experiment3, self.experiment4,
						self.experiment5, self.experiment6]

		if args.experiment != 0:
			experiments[args.experiment - 1]()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--find_optimal_param", action = "store_true")
    parser.add_argument("--experiment", type = int, default = 0, help="Experiment number")
    args = parser.parse_args()

    se = SingleExperiments()
    se.main(args)

