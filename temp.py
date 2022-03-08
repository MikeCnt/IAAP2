import pandas as pd
import xlrd
import random



def createRandomModel(nParameters):

	weights = []

	for i in range(nParameters):
		weights.append(random.randint(0,20))

	b = random.randint(0,50)

	return weights, b