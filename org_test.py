import pandas as pd
import numpy as np
import csv

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

with open('train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse) 
	fancy = data.corr()
	print(fancy)
	fancy.to_csv('correlation.csv')
	imp.fit([data['STARTING_LATITUDE']])
	imp.transform([data['STARTING_LATITUDE']])
	imp.fit([data['STARTING_LONGITUDE']])
	imp.transform([data['STARTING_LONGITUDE']])
	imp.fit([data['DESTINATION_LATITUDE']])
	imp.transform([data['DESTINATION_LATITUDE']])
	imp.fit([data['DESTINATION_LONGITUDE']])
	imp.transform([data['DESTINATION_LONGITUDE']])
	#data['STARTING_LATITUDE'].fillna(data['STARTING_LATITUDE'].interpolate(),inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	
	#data['STARTING_LONGITUDE'].fillna(data['STARTING_LONGITUDE'].interpolate(),inplace=True)
	#data['DESTINATION_LATITUDE'].fillna(data['DESTINATION_LATITUDE'].interpolate(),inplace=True)
	#data['DESTINATION_LONGITUDE'].fillna(data['DESTINATION_LONGITUDE'].interpolate(),inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
	data['WAIT_TIME'].fillna(0.0,inplace=True)
	#data['value'].fillna(data['value'].interpolate(),inplace=True)
	#print(data.isnull().sum())
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	#print(data.head())
	
	
	y = data['FARE']
	del data['FARE']
	del data['ID']
	X = data
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)
	

	
	#1.SVR
	"""
	from sklearn.svm import SVR
	reg = SVR(kernel='linear',C=5)
	reg.fit(X_train,y_train)
	prediction = reg.predict(X_test)
	print(r2_score(y_test,prediction))
	
	#2.RandomForestRegressor
	from sklearn.ensemble import RandomForestRegressor
	from sklearn import metrics, cross_validation
	reg1 =  RandomForestRegressor(random_state=15,n_estimators=350)
	reg1.fit(X ,y)
	prediction1 = reg1.predict(X_test)
	print("R r2_score",r2_score(y_test,prediction1))
	 	
	
	#3.KNeighborsRegressor
	from sklearn.neighbors import KNeighborsRegressor
	reg2 = KNeighborsRegressor(n_neighbors=49)
	reg2.fit(X ,y)	
	prediction2 = reg2.predict(X_test)
	print("KN",r2_score(y_test,prediction2))
	
	#4.linear_model
	from sklearn import linear_model
	reg3 = linear_model.LinearRegression()
	reg3.fit(X ,y)
	prediction3 = reg3.predict(X_test)
	print("LR",r2_score(y_test,prediction3))
	
	#5.linear_model	#Least Absolute Shrinkage and Selection Operator
	#if reg coeff are large lasso is used 
	from sklearn import linear_model
	reg4 = linear_model.Lasso(alpha=0.1)
	reg4.fit(X ,y)
	prediction4 = reg4.predict(X_test)
	print("Lasso",r2_score(y_test,prediction4))
	"""
	"""
	#6.GradientBoostingRegressor
	from sklearn import ensemble
	reg5 = ensemble.GradientBoostingRegressor( max_depth=10)
	reg5.fit(X ,y)
	prediction5 = reg5.predict(X_test)
	print("GBR",r2_score(y_test,prediction5))
	
	#7. DecisionTreeRegressor
	from sklearn.tree import DecisionTreeRegressor
	reg6 = DecisionTreeRegressor(max_depth=10)
	reg6.fit(X ,y)
	prediction6 = reg6.predict(X_test)
	print("DT",r2_score(y_test,prediction6))
	"""
	
	#8.XGBRegressor
	from xgboost.sklearn import XGBRegressor
	reg7 = XGBRegressor(learning_rate=0.1,max_estimators=400)
	reg7.fit(X ,y)
	prediction7 = reg7.predict(X_test)
	print("XGB",r2_score(y_test,prediction7))
	
	
	
	
