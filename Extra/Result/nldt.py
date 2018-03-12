import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))


with open('intracity_fare_train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse) 
	data['STARTING_LATITUDE'].fillna(data['STARTING_LATITUDE'].mean(),inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data['STARTING_LONGITUDE'].fillna(data['STARTING_LONGITUDE'].mean(),inplace=True)
	data['DESTINATION_LATITUDE'].fillna(data['DESTINATION_LATITUDE'].mean(),inplace=True)
	data['DESTINATION_LONGITUDE'].fillna(data['DESTINATION_LONGITUDE'].mean(),inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
	data['WAIT_TIME'].fillna(0.0,inplace=True)
	#print(data.isnull().sum())
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	#print(data.head())
	
	train_data = []						
	train_target = []

	#data.drop(data.str.contains("ID") == False)
	y = data['FARE']
	del data['FARE']
	del data['ID']
	X = data
	
	with open('intracity_fare_test.csv') as TestData:
		data1 = pd.read_csv(TestData,delimiter=',',parse_dates=True,date_parser=dateparse)
		data1['STARTING_LATITUDE'].fillna(data1['STARTING_LATITUDE'].mean(),inplace=True)
		data1['TIMESTAMP'] = pd.to_datetime(data1['TIMESTAMP'])
		data1['TIMESTAMP'] = (data1['TIMESTAMP'] - data1['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		data1['STARTING_LONGITUDE'].fillna(data1['STARTING_LONGITUDE'].mean(),inplace=True)
		data1['DESTINATION_LATITUDE'].fillna(data1['DESTINATION_LATITUDE'].mean(),inplace=True)
		data1['DESTINATION_LONGITUDE'].fillna(data1['DESTINATION_LONGITUDE'].mean(),inplace=True)
		data1['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
		data1['WAIT_TIME'].fillna(0.0,inplace=True)

		lbl1 = preprocessing.LabelEncoder()
		lbl1.fit(list(data1['VEHICLE_TYPE'].values))
		data1['VEHICLE_TYPE'] = lbl1.transform(list(data1['VEHICLE_TYPE'].values))

		del data1['ID']
		
		
		from sklearn.ensemble import RandomForestRegressor
		reg = RandomForestRegressor(max_depth=30, random_state=0)
		reg.fit(X ,y)	
		prediction = reg.predict(data1)
		#from sklearn.metrics import r2_score
		#print(r2_score(y_test,prediction))
		with open('result.csv',"w") as f:
			writer = csv.writer(f)
			ps = ['','FARE']
			writer.writerow(ps)
			
			x=1
			for Row in prediction:
				writer.writerow([x,Row])
				x += 1
			



	
	

