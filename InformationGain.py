from __future__ import print_function
import numpy as np 
import pandas as pd
import operator
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

class FeatureSelection(object):
	"""docstring for FeatureSelection"""
	def __init__(self, arg):
		self.arg = arg
		self.cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','y']
		self.aus = pd.read_csv(self.arg,names=self.cols)
		self.dis = [0,3,4,5,7,8,10,11]
		self.X = self.aus.iloc[:,0:14]
		self.Y = self.aus.iloc[:,14:15]
		#print (self.Y)

	def mutual_info_calculator(self):
		self.ans = mutual_info_regression(self.X,self.Y.values.ravel(),discrete_features=self.dis)
		self.ig = {}
		for i in range(14):
			self.ig[self.cols[i]]=self.ans[i]
		print (self.ig)
		self.ig = (sorted(self.ig.items(), key=operator.itemgetter(1),reverse=1))
		print (self.ig)

obj = FeatureSelection('australian.csv')
obj.mutual_info_calculator()
