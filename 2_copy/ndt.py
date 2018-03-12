import pandas as pd
import numpy as np
import csv
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))


with open('intracity_fare_train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse) 
	data['STARTING_LATITUDE'].fillna(data['STARTING_LATITUDE'].interpolate(),inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data['STARTING_LONGITUDE'].fillna(data['STARTING_LONGITUDE'].interpolate(),inplace=True)
	data['DESTINATION_LATITUDE'].fillna(data['DESTINATION_LATITUDE'].interpolate(),inplace=True)
	data['DESTINATION_LONGITUDE'].fillna(data['DESTINATION_LONGITUDE'].interpolate(),inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
	data['WAIT_TIME'].fillna(0.0,inplace=True)
	#data['FINAL'].fillna(data['FINAL'].mean(),inplace=True)
	data['DIST'].fillna(data['DIST'].mean(),inplace=True)
	#print(data.isnull().sum())
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	#print(data.head())
	
	
	y = data['FARE']
	del data['FARE']
	del data['DIST']
	del data['ID']
	X = data
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
	
	"""
	
	from sklearn.svm import SVR
	reg = SVR(kernel='linear',C=1)
	reg.fit(X_train,y_train)
	prediction = reg.predict(X_test)
	print(r2_score(y_test,prediction))
	"""
	from xgboost.sklearn import XGBRegressor
	reg7 = XGBRegressor(max_depth=7,learning_rate=0.1,n_estimators=400)
	reg7.fit(X ,y)
	prediction7 = reg7.predict(X_test)
	print("xgboost",r2_score(y_test,prediction7))
		
	
	from sklearn.ensemble import RandomForestRegressor
	reg1 = RandomForestRegressor(random_state=15,n_estimators=350)
	reg1.fit(X ,y)
	prediction1 = reg1.predict(X_test)
	print("R",r2_score(y_test,prediction1))
	
	from sklearn.neighbors import KNeighborsRegressor
	reg2 = KNeighborsRegressor(n_neighbors=49)
	reg2.fit(X ,y)	
	prediction2 = reg2.predict(X_test)
	print("KN",r2_score(y_test,prediction2))
	
	from sklearn import linear_model
	reg3 = linear_model.LinearRegression()
	reg3.fit(X ,y)
	prediction3 = reg3.predict(X_test)
	print("LR",r2_score(y_test,prediction3))
	
	from sklearn import linear_model
	reg4 = linear_model.Lasso(alpha=0.1)
	reg4.fit(X ,y)
	prediction4 = reg4.predict(X_test)
	print("Lasso",r2_score(y_test,prediction4))
	
	from sklearn import ensemble
	reg5 = ensemble.GradientBoostingRegressor(n_estimators=10, max_depth=8, learning_rate=1.0)
	reg5.fit(X ,y)
	prediction5 = reg5.predict(X_test)
	print("GBR",r2_score(y_test,prediction5))
	
	from sklearn.tree import DecisionTreeRegressor
	reg6 = DecisionTreeRegressor(max_depth=20,random_state=1)
	reg6.fit(X ,y)
	prediction6 = reg6.predict(X_test)
	print("DT",r2_score(y_test,prediction6))
	
	

	
	
