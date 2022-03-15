import csv
import pandas as pd
import random

def readDB():
	db = []

	with open('P2DB.csv') as file:
		reader = csv.reader(file, delimiter=',')
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
	yEstimatedVectorMAE = []
	for i in range(len(db)):
		temp = db[i]
		yEstimated = 0
		for j in range(len(temp)-1):
			yEstimated = yEstimated + (weights[j] * temp[j])
		
		yEstimated = yEstimated + b
		yEstimatedVectorMAE.append(yEstimated)
		sum = sum + abs(yEstimated - temp[len(temp)-1])

	error = sum / len(db)

	return error, yEstimatedVectorMAE

def errorFunctionMSE(weights, b, db):
	sum = 0
	yEstimatedVectorMSE = []
	for i in range(len(db)):
		temp = db[i]
		yEstimated = 0
		for j in range(len(temp)-1):
			yEstimated = yEstimated + (weights[j] * temp[j])
		
		yEstimated = yEstimated + b
		yEstimatedVectorMSE.append(yEstimated)
		sum = sum + pow(yEstimated - temp[len(temp)-1], 2)
	
	error = sum / len(db)

	return error, yEstimatedVectorMSE


def desGradientAdjustmentMAE(weights, b, db, u, yEstimatedMAE):

	q = len(db)
	for i in range(len(weights)):
		
		sum = 0
		for j in range(len(db)):
			temp = db[j]
			if yEstimatedMAE[j] - temp[len(temp)-1] < 0:
				for k in range(len(temp)-1):
					sum += (-1 * temp[k])
			else:
				for k in range(len(temp)-1):
					sum += (1 * temp[k])

		weights[i] -= (u/q) * sum

	sum = 0
	for i in range(len(db)):
		
		temp = db[i]
		if yEstimatedMAE[i] - temp[len(temp)-1] < 0:    
			sum += -1
		else:
			sum += 1

	b -= (u/q) * sum

	return weights, b


def main():

	dbParams = readDB()
	randomModel = createRandomModel(dbParams[1])

	print(randomModel[0])
	errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
	print(errorVectorMAE[0])

	lastError = errorVectorMAE[0]
	newError = lastError
	newModel = randomModel
	while newError <= lastError:
		lastError = newError
		newModel = desGradientAdjustmentMAE(newModel[0], newModel[1], dbParams[0], 0.1, errorVectorMAE[1])
		aux = errorFunctionMAE(newModel[0], newModel[1], dbParams[0])
		newError = aux[0]

	print("\n")
	print(newModel[0])
	print(newError)


if __name__ == "__main__":
	main()
