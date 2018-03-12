import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))


with open('train2.csv') as csvfile1,open('train1.csv') as csvfile2:
	data1 = pd.read_csv(csvfile1,delimiter=',',parse_dates=True,date_parser=dateparse)
	data2 = pd.read_csv(csvfile2,delimiter=',',parse_dates=True,date_parser=dateparse)
	
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
	
	#data1.hist()
	#plt.show()		
	fancy = data1.corr()
	fancy.to_csv('correlation1.csv')
			
	y1 = data1['FARE']
	del data1['FARE']
	del data1['ID']
	X1 = data1
	
	data2['STARTING_LATITUDE'].fillna(data2['STARTING_LATITUDE'].mean(),inplace=True)
	data2['TIMESTAMP'] = pd.to_datetime(data2['TIMESTAMP'])
	data2['TIMESTAMP'] = (data2['TIMESTAMP'] - data2['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data2['STARTING_LONGITUDE'].fillna(data2['STARTING_LONGITUDE'].mean(),inplace=True)
	data2['DESTINATION_LATITUDE'].fillna(data2['DESTINATION_LATITUDE'].mean(),inplace=True)
	data2['DESTINATION_LONGITUDE'].fillna(data2['DESTINATION_LONGITUDE'].mean(),inplace=True)
	
	
	lbl2 = preprocessing.LabelEncoder()
	lbl2.fit(list(data2['VEHICLE_TYPE'].values))
	data2['VEHICLE_TYPE'] = lbl2.transform(list(data2['VEHICLE_TYPE'].values))
			
	
	del data2['ID']
	del data2['TOTAL_LUGGAGE_WEIGHT']
	del data2['WAIT_TIME']
	X2 = data2
	
	#data2.hist()
	#plt.show()
	fancy = data2.corr()
	fancy.to_csv('correlation2.csv')
	
	y2 = data2['FARE']
	del data2['FARE']
	
	with open('test2.csv') as TestData1,open('test1.csv') as TestData2:
		data3 = pd.read_csv(TestData1,delimiter=',',parse_dates=True,date_parser=dateparse)
		data4 = pd.read_csv(TestData2,delimiter=',',parse_dates=True,date_parser=dateparse)
		
		data3['STARTING_LATITUDE'].fillna(data3['STARTING_LATITUDE'].mean(),inplace=True)
		data3['TIMESTAMP'] = pd.to_datetime(data3['TIMESTAMP'])
		data3['TIMESTAMP'] = (data3['TIMESTAMP'] - data3['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		data3['STARTING_LONGITUDE'].fillna(data3['STARTING_LONGITUDE'].mean(),inplace=True)
		data3['DESTINATION_LATITUDE'].fillna(data3['DESTINATION_LATITUDE'].mean(),inplace=True)
		data3['DESTINATION_LONGITUDE'].fillna(data3['DESTINATION_LONGITUDE'].mean(),inplace=True)
		data3['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
		data3['WAIT_TIME'].fillna(0.0,inplace=True)
	
		lbl3 = preprocessing.LabelEncoder()
		lbl3.fit(list(data3['VEHICLE_TYPE'].values))
		data3['VEHICLE_TYPE'] = lbl3.transform(list(data3['VEHICLE_TYPE'].values))
			
		y3 = data3['ID']
		del data3['ID']
		X3 = data3
	
		data4['STARTING_LATITUDE'].fillna(data4['STARTING_LATITUDE'].mean(),inplace=True)
		data4['TIMESTAMP'] = pd.to_datetime(data4['TIMESTAMP'])
		data4['TIMESTAMP'] = (data4['TIMESTAMP'] - data4['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		data4['STARTING_LONGITUDE'].fillna(data4['STARTING_LONGITUDE'].mean(),inplace=True)
		data4['DESTINATION_LATITUDE'].fillna(data4['DESTINATION_LATITUDE'].mean(),inplace=True)
		data4['DESTINATION_LONGITUDE'].fillna(data4['DESTINATION_LONGITUDE'].mean(),inplace=True)
	
	
		lbl4 = preprocessing.LabelEncoder()
		lbl4.fit(list(data4['VEHICLE_TYPE'].values))
		data4['VEHICLE_TYPE'] = lbl4.transform(list(data4['VEHICLE_TYPE'].values))
			
		y4 = data4['ID']
		del data4['ID']
		del data4['TOTAL_LUGGAGE_WEIGHT']
		del data4['WAIT_TIME']
		X4 = data4
		
		from xgboost.sklearn import XGBRegressor
		reg1 = XGBRegressor(learning_rate=0.1,max_depth=8)
		reg1.fit(X1 ,y1)
		
		reg2 =  XGBRegressor(learning_rate=0.1,max_depth=8)
		reg2.fit(X2 ,y2)
		#reg2 = ensemble.GradientBoostingRegressor( max_depth=9)
		#reg2.fit(X2 ,y2)
		
		prediction1 = reg1.predict(data3)
		prediction2 = reg2.predict(data4)
		
		with open('result1.csv',"w") as f:
			writer = csv.writer(f)
			#ps = ['ID','FARE']
			#writer.writerow(ps)
			
			for x,vc  in  zip(y3,prediction1):
				writer.writerow([x,vc])
				
		with open('result2.csv',"w") as f:
			writer = csv.writer(f)
			
			for x,vc  in  zip(y4,prediction2):
				writer.writerow([x,vc])
		
		
		
		
		
	
	

	
	

