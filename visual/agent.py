import numpy as np
from argparse import ArgumentParser
from nupic.algorithms.temporal_memory import TemporalMemory as TM
from nupic.algorithms.spatial_pooler import SpatialPooler
from parse_input import get_problems


def sp_compute(layer, input, learn=True):
	assert layer.getInputDimensions() == input.shape, "Wrong input size"
	output = np.zeros((layer.getNumColumns(), ), dtype="int")
	layer.compute(input, learn = learn, activeArray = output)
	return output


def tm_compute(layer, input, learn=True):
	assert layer.getColumnDimensions() == input.shape, "Wrong input size"
	layer.compute(input, learn = learn)
	output = layer.getActiveCells()
	return output


def bk_tm_compute(layer, input, learn=True):
	assert layer.getColumnDimensions() == input.shape, "Wrong input size"
	layer.compute(input, enableLearn = learn, enableInference=True)
	output = layer.getActiveCells()
	return output


def predict_method(layer):
	cells = layer.getPredictiveCells()
	print(layer.getColumnDimensions())
	print(cells)
	output = np.zeros(layer.getColumnDimensions())
	for cell in cells:
		output[cell] = 1
	return output


def linearize(window):
	(height, width) = window.shape
	return window.reshape((height * width, ))


def reverse_linearize(window):
	length = len(window)
	new_size = length ** 0.5
	return window.reshape((new_size, new_size))	


def run(layers, problems):
	output_windows = []
	num_iter = 20
	correct_predictions = np.zeros((len(problems), num_iter))
	for i in range(num_iter):
		#print '-------- ', i, '---------------'

		for j in range(len(problems)):
			problem = problems[j]
			#print('')
			#print(problem['Attributes']['result'])	
			res_idx = problem['Attributes']['result'] - 1
			
			'''
			for _ in range(10):
				for window in problem['Input'] + [problem['Output'][res_idx]]:
					last_input = window
					for (layer, compute_method) in layers:
						last_input = compute_method(layer, last_input, True)
			'''

			for window in problem['Input'][:2]:
				last_input = linearize(window)
				#print(last_input.shape)
				for (layer, compute_method) in layers:
					last_input = compute_method(layer, last_input, True)

			last_input = linearize(problem['Input'][2])
			for (layer, compute_method) in layers:
				last_input = compute_method(layer, last_input, True)
			
			(last_layer, compute_method) = layers[-1]
			predict = predict_method(last_layer)
			
			results = []
			for k in range(len(problem['Output'])):
				last_input = linearize(problem['Output'][k])
				for (layer, compute_method) in layers[:-1]:
					last_input = compute_method(layer, last_input, False)
				active = np.sum(last_input)
				overlap = np.sum(predict * last_input)
				if active == 0:
					ratio = 0
				else:
					ratio = float(overlap)/active
					results += [ratio]
				'''
				print 'Active: %d Overlap: %d Ratio: %0.3f' % (active, overlap, ratio),
				if i == res_idx:
					print('*')
				else:
					print('')
				'''

			if results[res_idx] == max(results):
				correct_predictions[j][i] = results[res_idx]
			
			last_input = linearize(problem['Output'][res_idx])
			#last_input = linearize(np.ones(problem['Input'][0].shape))
			#last_input = predict
			#print(last_input.shape)
			for (layer, compute_method) in layers:
				last_input = compute_method(layer, last_input, True)
			
			layers[-1][0].reset()
	print(correct_predictions)

def main(args):
	folder_name = '../Problems'
	problems = get_problems(folder_name)

	(h, w) = problems[0]['Input'][0].shape
	#dim1 = (h, w)
	#dim2 = (h / 2, w / 2)

	numColls1 = h * w
	numColls2 = numColls1
	#numColls3 = numColls2 / 2
	#numColls4 = numColls3 / 2

	sp1 = SpatialPooler(inputDimensions = (numColls1, ), columnDimensions = (numColls2, ), numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
	#sp2 = SpatialPooler(inputDimensions = (numColls1, ), columnDimensions = (numColls2, ), numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
	#sp3 = SpatialPooler(inputDimensions = (numColls2, ), columnDimensions = (numColls3, ), numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
	#sp4 = SpatialPooler(inputDimensions = (numColls3, ), columnDimensions = (numColls4, ), numActiveColumnsPerInhArea = -1, localAreaDensity = 0.05)
	'''
	bckTM = BacktrackingTM(numberOfCols=numColls2, cellsPerColumn=2,
						initialPerm=0.5, connectedPerm=0.5,
						minThreshold=10, newSynapseCount=10,
						permanenceInc=0.1, permanenceDec=0.0,
						activationThreshold=8,
						globalDecay=0, burnIn=1,
						checkSynapseConsistency=False,
						pamLength=10)
	'''
	tm = TM(columnDimensions=(numColls2, ), cellsPerColumn=2,
						initialPermanence=0.5, connectedPerm=0.5,
						minThreshold=10, newSynapseCount=10,
						permanenceInc=0.1, permanenceDec=0.0,
						activationThreshold=10,
						globalDecay=0, burnIn=1,
						checkSynapseConsistency=False,
						pamLength=10)
	layers = [(sp1, sp_compute), 
			  (tm, tm_compute)]	
	'''
	layers = [(sp1, compute_spatial), 
			  (sp2, compute_spatial), 
			  (sp3, compute_spatial),
			  (sp4, compute_spatial),
			  (bckTM, compute_temporal)]	
	'''
	run(layers, problems)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
	
