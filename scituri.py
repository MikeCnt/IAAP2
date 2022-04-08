import csv
import pandas as pd
import random
import copy
import turicreate as tc
import sklearn 
import numpy
from sklearn import linear_model


def main():

  #Turi Create
  datatc =  tc.SFrame.read_csv('P2DB(Diabetes)2.csv')
  
  
  model_ridge = tc.linear_regression.create(datatc, 'BP', l2_penalty=0.1, max_iterations= 1000)
  model_lasso = tc.linear_regression.create(datatc, 'BP', l2_penalty=0., l1_penalty=1.0, max_iterations= 1000)
  model_enet  = tc.linear_regression.create(datatc, 'BP', l2_penalty=0.5, l1_penalty=0.5, max_iterations= 1000)

  #SciKit Learn
  datapd = pd.read_csv('P2DB(Diabetes)2.csv')
  datask = pd.DataFrame.to_numpy(datapd)
  y = numpy.arange(442)
  regLasso = linear_model.Lasso(alpha = 1, random_state=10).fit(datask, y)
  print (regLasso.score(datask, y))

  regRidge = linear_model.Ridge(alpha = 1, random_state=10).fit(datask, y)
  print (regRidge.score(datask, y))
  
  regElasticNet = linear_model.ElasticNet(alpha = 1, random_state=10).fit(datask, y)
  print (regElasticNet.score(datask, y))


if __name__ == "__main__":
	main()