import pandas as pd
import numpy as np
import csv
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
	
	data['STARTING_LATITUDE'].fillna(imp.transform(data['STARTING_LATITUDE']),inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data['STARTING_LONGITUDE'].fillna(np.nan,inplace=True)
	data['DESTINATION_LATITUDE'].fillna(np.nan,inplace=True)
	data['DESTINATION_LONGITUDE'].fillna(np.nan,inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(np.nan,inplace=True)
	data['WAIT_TIME'].fillna(np.nan,inplace=True)
	#print(data.isnull().sum())
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	

	del data['ID']
	#data.hist()
	#data.plot(kind='density', subplots=True, layout=(4,3), sharex=False)
	
	y = data['FARE']
	del data['FARE']
	
	X = data
	
	
	with open('test.csv') as TestData:
		data1 = pd.read_csv(TestData,delimiter=',',parse_dates=True,date_parser=dateparse)
		
		data1['STARTING_LATITUDE'].fillna(np.nan,inplace=True)
		data1['TIMESTAMP'] = pd.to_datetime(data1['TIMESTAMP'])
		data1['TIMESTAMP'] = (data1['TIMESTAMP'] - data1['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		data1['STARTING_LONGITUDE'].fillna(np.nan,inplace=True)
		data1['DESTINATION_LATITUDE'].fillna(np.nan,inplace=True)
		data1['DESTINATION_LONGITUDE'].fillna(np.nan,inplace=True)
		data1['TOTAL_LUGGAGE_WEIGHT'].fillna(np.nan,inplace=True)
		data1['WAIT_TIME'].fillna(np.nan,inplace=True)
		
		lbl1 = preprocessing.LabelEncoder()
		lbl1.fit(list(data1['VEHICLE_TYPE'].values))
		data1['VEHICLE_TYPE'] = lbl1.transform(list(data1['VEHICLE_TYPE'].values))
		
	
		del data1['ID']
		
		#from sklearn import ensemble
		#reg = ensemble.GradientBoostingRegressor(max_depth=6)
		from xgboost.sklearn import XGBRegressor
		reg =  XGBRegressor(learning_rate=0.1,max_depth=24,min_child_weight= 7,subsample = 0.7, colsample_bytree= 0.7,objective= 'reg:linear')
		reg.fit(X ,y)	
		prediction = reg.predict(data1)
		#from sklearn.metrics import r2_score
		#print(r2_score(y_test,prediction))
		with open('result.csv',"w") as f:
			writer = csv.writer(f)
			ps = ['ID','FARE']
			writer.writerow(ps)
			
			x=1
			for Row in prediction:
				writer.writerow([x,Row])
				x += 1
			



	
	

