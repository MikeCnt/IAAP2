import csv
import pandas as pd
import xlrd
import random

def readDB():
    db = []

    with open('P2DB.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            db.append(row)
    
    nParameters = len(db[0])

    return db, nParameters

def createRandomModel(nParameters):
	weights = []

	for i in range(nParameters):
		weights.append(random.randint(0,20))

	b = random.randint(-20,20)

	return weights, b


def errorFunctionMAE(weights, b, db):
    sum = 0

    for i in range(len(db)):
        temp = db[i+1]
        for j in range(len(temp)-1):
            yEstimated = 0
            yEstimated = yEstimated + (weights[j] * temp[j])
        
        yEstimated = yEstimated + b
        sum = sum + abs(yEstimated - temp[len(temp)-1])

    error = sum / len(db)

    return error

def errorFunctionMSE(weights, b, db):
    sum = 0

    for i in range(len(db)):
        temp = db[i+1]
        for j in range(len(temp)-1):
            yEstimated = 0
            yEstimated = yEstimated + (weights[j] * temp[j])
        
        yEstimated = yEstimated + b
        sum = sum + pow(yEstimated - temp[len(temp)-1], 2)
    
    error = sum / len(db)

    return error
        
## YA QUE TENEMOS LAS FUNCIONES DEL ERROR
## LO QUE TENEMOS QUE HACER AHORA ES MINIMIZAR EL ERROR
## CON EL METODO DEL GRADIENTE, ACTUALIZANDO PESOS Y SESGO
        
def main():
    nParameters = readDB()
    createRandomModel(nParameters)

if __name__ == "__main__":
    main()
