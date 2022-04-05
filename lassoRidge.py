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

	return error, yEstimatedVector

	

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

	return error, yEstimatedVector


def errorLasso(weights, error, coefAprend):

	L1 = 0

	for i in range(len(weights)):

		L1 += abs(weights[i])

	errorLasso = error + (coefAprend * L1)

	return errorLasso


def errorRidge(weights, error, coefAprend):
	
	L2 = 0

	for i in range(len(weights)):

		L2 += pow(weights[i], 2)

	errorRidge = error + (coefAprend * L2)

	return errorRidge


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
	termRegular = 0.01



##	ERRORES MAE

	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])


#	ERROR MAE
	errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error (MAE) -- ", errorVectorMAE[0])

	lastError = copy.deepcopy(errorVectorMAE[0])

	newError = copy.deepcopy(lastError)

	newModel = copy.deepcopy(randomModel)


#	ERROR LASSO MAE
	errorL = errorLasso(randomModel[0], errorVectorMAE[0], termRegular)
	print("Random Model Error Lasso (MAE) -- ", errorL)

	lastErrorLasso = copy.deepcopy(errorL)

	newErrorLasso = copy.deepcopy(lastErrorLasso)


#	ERROR RIDGE MAE
	errorR = errorRidge(randomModel[0], errorVectorMAE[0], termRegular)
	print("Random Model Error Ridge (MAE) -- ", errorR)

	lastErrorRidge = copy.deepcopy(errorR)

	newErrorRidge = copy.deepcopy(lastErrorRidge)


#	AJUSTE
	while newError <= lastError:
		lastError = copy.deepcopy(newError)
		lastErrorLasso = copy.deepcopy(newErrorLasso)
		lastErrorRidge = copy.deepcopy(newErrorRidge)
		newModel = desGradientAdjustmentMAE(newModel[0], newModel[1], dbParams[0], coefAprendMAE, errorVectorMAE[1])	
		aux = errorFunctionMAE(newModel[0], newModel[1], dbParams[0])
		auxL = errorLasso(newModel[0], aux[0], termRegular)
		auxR = errorRidge(newModel[0], aux[0], termRegular)
		newError = copy.deepcopy(aux[0])
		newErrorLasso = copy.deepcopy(auxL)
		newErrorRidge = copy.deepcopy(auxR)

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])


#	ERROR MAE
	print("Adjust Model Error (MAE) -- ", lastError)


#	ERROR LASSO MAE
	print("Adjust Model Error Lasso (MAE) -- ", lastErrorLasso)


#	ERROR RIDGE MAE
	print("Adjust Model Error Ridge (MAE) -- ", lastErrorRidge)
	print("<================================================================>")
	





#	ERRORES MSE
	print("\n")
	print("\n")
	print("<================================================================>")
	print("Random Model Weigths     -- ", randomModel[0])
	print("Random Model Bias        -- ", randomModel[1])


#	ERROR MSE
	errorVectorMSE = errorFunctionMSE(randomModel[0], randomModel[1], dbParams[0])
	print("Random Model Error (MSE) -- ", errorVectorMSE[0])

	lastError = copy.deepcopy(errorVectorMSE[0])

	newError = copy.deepcopy(lastError)

	newModel = copy.deepcopy(randomModel)


#	ERROR LASSO MSE
	errorL = errorLasso(randomModel[0], errorVectorMSE[0], termRegular)
	print("Random Model Error Lasso (MSE) -- ", errorL)

	lastErrorLasso = copy.deepcopy(errorL)

	newErrorLasso = copy.deepcopy(lastErrorLasso)


#	ERROR RIDGE MSE
	errorR = errorRidge(randomModel[0], errorVectorMSE[0], termRegular)
	print("Random Model Error Ridge (MSE) -- ", errorR)

	lastErrorRidge = copy.deepcopy(errorR)

	newErrorRidge = copy.deepcopy(lastErrorRidge)


#	AJUSTE
	while newError <= lastError:
		lastError = copy.deepcopy(newError)
		lastErrorLasso = copy.deepcopy(newErrorLasso)
		lastErrorRidge = copy.deepcopy(newErrorRidge)
		newModel = desGradientAdjustmentMSE(newModel[0], newModel[1], dbParams[0], coefAprendMSE, errorVectorMSE[1])	
		aux = errorFunctionMSE(newModel[0], newModel[1], dbParams[0])
		auxL = errorLasso(newModel[0], aux[0], termRegular)
		auxR = errorRidge(newModel[0], aux[0], termRegular)
		newError = copy.deepcopy(aux[0])
		newErrorLasso = copy.deepcopy(auxL)
		newErrorRidge = copy.deepcopy(auxR)

	print("\n")
	print("Adjust Model Weigths     -- ", newModel[0])
	print("Adjust Model Bias        -- ", newModel[1])


#	ERROR MSE
	print("Adjust Model Error (MSE) -- ", lastError)


#	ERROR LASSO MSE
	print("Adjust Model Error Lasso (MSE) -- ", lastErrorLasso)


#	ERROR RIDGE MSE
	print("Adjust Model Error Ridge (MSE) -- ", lastErrorRidge)
	print("<================================================================>")







if __name__ == "__main__":
	main()
