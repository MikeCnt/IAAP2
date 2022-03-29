import csv
from numpy import real_if_close
import pandas as pd
import random
import copy


def readDB():
	db = [] 

	with open('P2DB(Diabetes).csv') as file:
		reader = csv.reader(file, delimiter='\t')
		lines = 0
		for row in reader:
			if lines == 0:
				lines = lines + 1
			else:
				for i in range(len(row)):
					row[i] = float(row[i])

				db.append(row)
		
		nParameters = len(db[0]) - 1

	return db, nParameters



def createRandomModel(nParameters):
	weights = []
	for i in range(nParameters):
		weights.append(random.randint(0,20))
	
	b = random.randint(-10,10)

	return weights, b


def errorFunctionMAE(weights, b, db):
	sum = 0
	yEstimatedVector = []
	for i in range(len(db)):
		temp = db[i]
		yEstimated = 0
		for j in range(len(temp)-1):
			yEstimated = yEstimated + (weights[j] * temp[j])
		
		yEstimated = yEstimated + b
		yEstimatedVector.append(yEstimated)
		sum = sum + abs(yEstimated - temp[len(temp)-1])

	error = sum / len(db)

	lassoSum = 0
	ridgeSum = 0

	for i in range(len(weights)):

		lassoSum += abs(weights[i])
		ridgeSum += pow(weights[i], 2)

	paramRegularization = 0.01

	errorLasso = error + paramRegularization * lassoSum
	errorRidge = error + paramRegularization * ridgeSum

	return error, yEstimatedVector, errorLasso, errorRidge

	

def errorFunctionMSE(weights, b, db):
	sum = 0
	yEstimatedVector = []
	for i in range(len(db)):
		temp = db[i]
		yEstimated = 0
		for j in range(len(temp)-1):
			yEstimated = yEstimated + (weights[j] * temp[j])
		
		yEstimated = yEstimated + b
		yEstimatedVector.append(yEstimated)
		sum = sum + pow((yEstimated - temp[len(temp)-1]), 2)
	
	error = sum / len(db)

	lassoSum = 0
	ridgeSum = 0

	for i in range(len(weights)):

		lassoSum += abs(weights[i])
		ridgeSum += pow(weights[i], 2)

	paramRegularization = 0.01

	errorLasso = error + paramRegularization * lassoSum
	errorRidge = error + paramRegularization * ridgeSum

	return error, yEstimatedVector, errorLasso, errorRidge


def desGradientAdjustmentMAE(weights, b, db, u, yEstimated):

	q = len(db)
	for i in range(len(weights)):
		
		sum = 0
		for j in range(len(db)):
			temp = db[j]
			if yEstimated[j] - temp[len(temp)-1] < 0:
				for k in range(len(temp)-1):
					sum += (-1 * temp[k])
			else:
				for k in range(len(temp)-1):
					sum += (1 * temp[k])

		weights[i] -= ((u/q) * sum)

	sum = 0
	for i in range(len(db)):
		
		temp = db[i]
		if yEstimated[i] - temp[len(temp)-1] < 0:    
			sum += -1
		else:
			sum += 1

	b -= (u/q) * sum

	return weights, b


def desGradientAdjustmentMSE(weights, b, db, u, yEstimated):

	q = len(db)
	for i in range(len(weights)):
		
		sum = 0
		for j in range(len(db)):
			temp = db[j]
			for k in range(len(temp)-1):
				sum += temp[k] * (yEstimated[j] - temp[len(temp)-1])

		weights[i] -= ((u/q) * sum)

	sum = 0
	for i in range(len(db)):
		
		temp = db[i]
		sum += (yEstimated[i] - temp[len(temp)-1]) 

	b -= (u/q) * sum

	return weights, b


def main():

	dbParams = readDB()
	randomModel = createRandomModel(dbParams[1])
	coefAprendMAE = 0.0001
	coefAprendMSE = 0.0000001



#	ERRORES MAE

##	ERROR MAE

	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error (MAE) -- ", errorVectorMAE[0])

	lastError = copy.deepcopy(errorVectorMAE[0])

	newError = copy.deepcopy(lastError)

	newModel = copy.deepcopy(randomModel)

	while newError <= lastError:
		lastError = copy.deepcopy(newError)
		newModel = desGradientAdjustmentMAE(newModel[0], newModel[1], dbParams[0], coefAprendMAE, errorVectorMAE[1])	
		aux = errorFunctionMAE(newModel[0], newModel[1], dbParams[0])
		newError = copy.deepcopy(aux[0])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error (MAE) -- ", lastError)
	print("<================================================================>")



##	ERROR LASSO MAE

	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error Lasso (MAE) -- ", errorVectorMAE[2])

	lastErrorLasso = copy.deepcopy(errorVectorMAE[2])

	newErrorLasso = copy.deepcopy(lastErrorLasso)

	newModel = copy.deepcopy(randomModel)

	while newErrorLasso <= lastErrorLasso:
		lastErrorLasso = copy.deepcopy(newErrorLasso)
		newModel = desGradientAdjustmentMAE(newModel[0], newModel[1], dbParams[0], coefAprendMAE, errorVectorMAE[1])	
		aux = errorFunctionMAE(newModel[0], newModel[1], dbParams[0])
		newErrorLasso = copy.deepcopy(aux[2])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error Lasso (MAE) -- ", lastErrorLasso)
	print("<================================================================>")



##	ERROR RIDGE MAE

	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error Ridge (MAE) -- ", errorVectorMAE[3])

	lastErrorRidge = copy.deepcopy(errorVectorMAE[3])

	newErrorRidge = copy.deepcopy(lastErrorRidge)

	newModel = copy.deepcopy(randomModel)

	while newErrorRidge <= lastErrorRidge:
		lastErrorRidge = copy.deepcopy(newErrorRidge)
		newModel = desGradientAdjustmentMAE(newModel[0], newModel[1], dbParams[0], coefAprendMAE, errorVectorMAE[1])	
		aux = errorFunctionMAE(newModel[0], newModel[1], dbParams[0])
		newErrorRidge = copy.deepcopy(aux[3])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error Ridge (MAE) -- ", lastErrorRidge)
	print("<================================================================>")
	





#	ERRORES MSE

##	ERROR MSE

	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMSE = errorFunctionMSE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error (MSE) -- ", errorVectorMSE[0])

	lastError = copy.deepcopy(errorVectorMSE[0])
	newError = copy.deepcopy(lastError)
	newModel = copy.deepcopy(randomModel)

	while newError <= lastError:
		lastError = copy.deepcopy(newError)
		newModel = desGradientAdjustmentMSE(newModel[0], newModel[1], dbParams[0], coefAprendMSE, errorVectorMSE[1])
		aux = errorFunctionMSE(newModel[0], newModel[1], dbParams[0])
		newError = copy.deepcopy(aux[0])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error (MSE) -- ", lastError)
	print("<================================================================>")



##	ERROR LASSO MSE

	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMSE = errorFunctionMSE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error Lasso (MSE) -- ", errorVectorMSE[2])

	lastErrorLasso = copy.deepcopy(errorVectorMSE[2])

	newErrorLasso = copy.deepcopy(lastErrorLasso)

	newModel = copy.deepcopy(randomModel)

	while newErrorLasso <= lastErrorLasso:
		lastErrorLasso = copy.deepcopy(newErrorLasso)
		newModel = desGradientAdjustmentMSE(newModel[0], newModel[1], dbParams[0], coefAprendMSE, errorVectorMSE[1])	
		aux = errorFunctionMSE(newModel[0], newModel[1], dbParams[0])
		newErrorLasso = copy.deepcopy(aux[2])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error Lasso (MSE) -- ", lastErrorLasso)
	print("<================================================================>")



##	ERROR RIDGE MSE

	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])
	errorVectorMSE = errorFunctionMSE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error Ridge (MSE) -- ", errorVectorMSE[3])

	lastErrorRidge = copy.deepcopy(errorVectorMSE[3])

	newErrorRidge = copy.deepcopy(lastErrorRidge)

	newModel = copy.deepcopy(randomModel)

	while newErrorRidge <= lastErrorRidge:
		lastErrorRidge = copy.deepcopy(newErrorRidge)
		newModel = desGradientAdjustmentMSE(newModel[0], newModel[1], dbParams[0], coefAprendMSE, errorVectorMSE[1])	
		aux = errorFunctionMSE(newModel[0], newModel[1], dbParams[0])
		newErrorRidge = copy.deepcopy(aux[3])

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])
	print("Adjust Model Error Ridge (MSE) -- ", lastErrorLasso)
	print("<================================================================>")
	print("<================================================================>")







if __name__ == "__main__":
	main()
