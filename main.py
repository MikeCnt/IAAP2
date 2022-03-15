import csv
from distutils.dep_util import newer
import pandas as pd
import xlrd
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
        
        nParameters = len(db[0])

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
        for j in range(len(temp)-1):
            yEstimated = 0
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
        for j in range(len(temp)-1):
            yEstimated = 0
            yEstimated = yEstimated + (weights[j] * temp[j])
        
        yEstimated = yEstimated + b
        yEstimatedVectorMSE.append(yEstimated)
        sum = sum + pow(yEstimated - temp[len(temp)-1], 2)
    
    error = sum / len(db)

    return error, yEstimatedVectorMSE
        
## YA QUE TENEMOS LAS FUNCIONES DEL ERROR
## LO QUE TENEMOS QUE HACER AHORA ES MINIMIZAR EL ERROR
## CON EL METODO DEL GRADIENTE, ACTUALIZANDO PESOS Y SESGO
## def desGradientAdjustment(weights, b, db, error, u):

## DESPUES DE TENER AMBOS ERRORES OPTIMIZADOS HAY QUE HACER COMPARACIONES
## def compareERROR(errorMAE, errorMSE):
        
def desGradientAdjustmentMAE(weights, b, db, u, yEstimated):
    for i in range(len(weights)):
        for j in range(len(db)):
            temp = db[j]
            sum = 0
            if yEstimated[j] - temp[len(temp)-1] < 0:
                for k in range(len(temp)-1):
                    sum = sum + (temp[k] * -1)
            else:
                for k in range(len(temp)-1):
                    sum = sum + (temp[k] * 1)

        weights[i] = weights[i] - ((u/len(db)) * sum)
    
    newError = errorFunctionMAE(weights, b, db)
    return newError[0]


def main():
    dbParams = []

    dbParams = readDB()
    randomModel = createRandomModel(dbParams[1])
    errorVectorMAE = errorFunctionMAE(randomModel[0], randomModel[1], dbParams[0])
    errorVectorMSE = errorFunctionMSE(randomModel[0], randomModel[1], dbParams[0])
    print(errorVectorMAE[0])
    print(errorVectorMSE[0])
    print("\n")
    
    for i in range(1000):
        newError = desGradientAdjustmentMAE(randomModel[0], randomModel[1], dbParams[0], 0.001, errorVectorMAE[1])

    print(newError)


if __name__ == "__main__":
    main()
