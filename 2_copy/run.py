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
	data['STARTING_LATITUDE'].fillna(np.nan,inplace=True)
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D')
	data['STARTING_LONGITUDE'].fillna(data['STARTING_LONGITUDE'].mean(),inplace=True)
	data['DESTINATION_LATITUDE'].fillna(data['DESTINATION_LATITUDE'].mean(),inplace=True)
	data['DESTINATION_LONGITUDE'].fillna(data['DESTINATION_LONGITUDE'].mean(),inplace=True)
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
	data['WAIT_TIME'].fillna(0.0,inplace=True)
	"""	
	data['LAT'].fillna(data['LAT'].mean(),inplace=True)
	data['LONG'].fillna(data['LONG'].mean(),inplace=True)
	data['SQRT LAT'].fillna(data['SQRT LAT'].mean(),inplace=True)
	data['SQRT LONG'].fillna(data['SQRT LONG'].mean(),inplace=True)
	data['ADDED'].fillna(data['ADDED'].mean(),inplace=True)
	"""	
	#data['FINAL'].fillna(data['FINAL'].mean(),inplace=True)
	data['DIST'].fillna(data['DIST'].mean(),inplace=True)
	
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(data['VEHICLE_TYPE'].values))
	data['VEHICLE_TYPE'] = lbl.transform(list(data['VEHICLE_TYPE'].values))
	
	del data['ID']
	fancy = data.corr()
	fancy.to_csv('correlation1.csv')
	y = data['FARE']
	del data['FARE']
	"""	
	del data['LAT']
	del data['LONG']
	del data['SQRT LAT']
	del data['SQRT LONG']
	del data['ADDED']
	"""	
	
	X = data
	
	
	with open('intracity_fare_test.csv') as TestData:
		data1 = pd.read_csv(TestData,delimiter=',',parse_dates=True,date_parser=dateparse)
		
		data1['STARTING_LATITUDE'].fillna(np.nan,inplace=True)
		data1['TIMESTAMP'] = pd.to_datetime(data1['TIMESTAMP'])
		data1['TIMESTAMP'] = (data1['TIMESTAMP'] - data1['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		data1['STARTING_LONGITUDE'].fillna(data1['STARTING_LONGITUDE'].mean(),inplace=True)
		data1['DESTINATION_LATITUDE'].fillna(data1['DESTINATION_LATITUDE'].mean(),inplace=True)
		data1['DESTINATION_LONGITUDE'].fillna(data1['DESTINATION_LONGITUDE'].mean(),inplace=True)
		data1['TOTAL_LUGGAGE_WEIGHT'].fillna(0.0,inplace=True)
		data1['WAIT_TIME'].fillna(0.0,inplace=True)
		"""		
		data1['LAT'].fillna(data1['LAT'].mean(),inplace=True)
		data1['LONG'].fillna(data1['LONG'].mean(),inplace=True)
		data1['SQRT LAT'].fillna(data1['SQRT LAT'].mean(),inplace=True)
		data1['SQRT LONG'].fillna(data1['SQRT LONG'].mean(),inplace=True)
		data1['ADDED'].fillna(data1['ADDED'].mean(),inplace=True)
		"""		
		#data1['FINAL'].fillna(data1['FINAL'].mean(),inplace=True)
		data1['DIST'].fillna(data1['DIST'].mean(),inplace=True)
		lbl1 = preprocessing.LabelEncoder()
		lbl1.fit(list(data1['VEHICLE_TYPE'].values))
		data1['VEHICLE_TYPE'] = lbl1.transform(list(data1['VEHICLE_TYPE'].values))
		
	
		del data1['ID']
		"""		
		del data1['LAT']
		del data1['LONG']
		del data1['SQRT LAT']
		del data1['SQRT LONG']
		del data1['ADDED']
		"""		
		
		
		#from sklearn import ensemble
		#reg = ensemble.GradientBoostingRegressor(max_depth=6)
		from sklearn.ensemble import RandomForestRegressor
		reg = RandomForestRegressor(max_depth=25, random_state=1)
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
			



	
	

