import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
import datetime
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

from sklearn.preprocessing import Imputer					#to handle missing data  using imputer function
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)	



with open('train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',',parse_dates=True,date_parser=dateparse)  #creating a pandas dataframe
	
	#data['STARTING_LATITUDE'].fillna(imp.transform(data['STARTING_LATITUDE']),inplace=True)
	#data['STARTING_LONGITUDE'].fillna(np.nan,inplace=True)
	#data['DESTINATION_LATITUDE'].fillna(np.nan,inplace=True)
	#data['DESTINATION_LONGITUDE'].fillna(np.nan,inplace=True)
	
	data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])  #converting timestamp to datetime datatype
	data['TIMESTAMP'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min())  / np.timedelta64(1,'D') #np.timedelta complements datetime
	data['TOTAL_LUGGAGE_WEIGHT'].fillna(np.nan,inplace=True)
	data['WAIT_TIME'].fillna(np.nan,inplace=True)
	
	
	imp.fit([data['STARTING_LATITUDE']])
	imp.transform([data['STARTING_LATITUDE']])
	imp.fit([data['STARTING_LONGITUDE']])
	imp.transform([data['STARTING_LONGITUDE']])
	imp.fit([data['DESTINATION_LATITUDE']])
	imp.transform([data['DESTINATION_LATITUDE']])
	imp.fit([data['DESTINATION_LONGITUDE']])
	imp.transform([data['DESTINATION_LONGITUDE']])
		
		
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
		
		#data1['STARTING_LATITUDE'].fillna(np.nan,inplace=True)
		data1['TIMESTAMP'] = pd.to_datetime(data1['TIMESTAMP'])
		data1['TIMESTAMP'] = (data1['TIMESTAMP'] - data1['TIMESTAMP'].min())  / np.timedelta64(1,'D')
		#data1['STARTING_LONGITUDE'].fillna(np.nan,inplace=True)
		#data1['DESTINATION_LATITUDE'].fillna(np.nan,inplace=True)
		#data1['DESTINATION_LONGITUDE'].fillna(np.nan,inplace=True)
		data1['TOTAL_LUGGAGE_WEIGHT'].fillna(np.nan,inplace=True) #filling missing values with np.nan
		data1['WAIT_TIME'].fillna(np.nan,inplace=True)			   #filling missing values with np.nan
		
		imp.fit([data1['STARTING_LATITUDE']])					#imputing mising values with mean 
		imp.transform([data1['STARTING_LATITUDE']])
		imp.fit([data1['STARTING_LONGITUDE']])
		imp.transform([data1['STARTING_LONGITUDE']])
		imp.fit([data1['DESTINATION_LATITUDE']])
		imp.transform([data1['DESTINATION_LATITUDE']])
		imp.fit([data1['DESTINATION_LONGITUDE']])
		imp.transform([data1['DESTINATION_LONGITUDE']])
		
		lbl1 = preprocessing.LabelEncoder()			#using label encoder to handle strings in the data
		lbl1.fit(list(data1['VEHICLE_TYPE'].values))
		data1['VEHICLE_TYPE'] = lbl1.transform(list(data1['VEHICLE_TYPE'].values))
		
	
		del data1['ID']			#deleting id column
		
	
		from xgboost.sklearn import XGBRegressor			#using xgbregressor 
		reg =  XGBRegressor(learning_rate=0.0398,max_depth=7)	#learning rate makes the model robust by shrinking weights on each step
		reg.fit(X ,y)										#higher depth allows the model to learn relations specific to particular sample
		prediction = reg.predict(data1)
		
		with open('result.csv',"w") as f:
			writer = csv.writer(f)							#returns a writer object 
			ps = ['ID','FARE']
			writer.writerow(ps)							#writing row wise 
			
			x=1
			for Row in prediction:
				writer.writerow([x,Row])
				x += 1
			



	
	

