import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))


with open('train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse) 
	data['STARTING_LATITUDE'].fillna(data['STARTING_LATITUDE'].mean(),inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data['STARTING_LONGITUDE'].fillna(data['STARTING_LONGITUDE'].mean(),inplace=True)
	data['DESTINATION_LATITUDE'].fillna(data['DESTINATION_LATITUDE'].mean(),inplace=True)
	data['DESTINATION_LONGITUDE'].fillna(data['DESTINATION_LONGITUDE'].mean(),inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
	data['WAIT_TIME'].fillna(0.0,inplace=True)
	print(data.isnull().sum())
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	print(data.head())
	
	train_data = []						#for features
	train_target = []

	#data.drop(data.str.contains("ID") == False)
	y = data['FARE']
	del data['FARE']
	
	del data['ID']
	X = data
	
	with open('test.csv') as Testdata:
		
	
	
	from sklearn.svm import SVR
	reg = SVR(kernel='linear',C=1)
	reg.fit(X,y)
	prediction = reg.predict(X_test)
	from sklearn.metrics import r2_score
	print(r2_score(y_test,prediction))
	
	#print(data)
