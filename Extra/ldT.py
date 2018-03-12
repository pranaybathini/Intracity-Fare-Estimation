import pandas as pd
import numpy as np
import csv
from sklearn import linear_model


with open('train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    	#reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	dic = {'ac bus':100.0,'auto rickshaw':40.0,'mini bus':30.0,'taxi non ac':50.0,'taxi ac':80.0,'bus':60.0,'metro':90.0,'vehicle_type':0}
	
	
	for row in data:
		 l1 = row[2:6]
		 l1 = [float(i) for i in l1]
		 l3  = row[7:11]
		 l3 = [float(j) for j in l3]
		 train_data.append(l3)
		 #[dic[row[6].lower()]]
		# train_data.append(l1)
		# train_data.append(row[7:11])
		#lc = row[6].lower()
		#train_data.append(dic[lc])
		#train_data.append(row[7:11])
		 train_target.append(float(row[11]))
	#test_idx = [0]
	#train_data = np.delete(train_data,test_idx,axis=0)
	#train_target = np.delete(train_target,test_idx)
	#print(train_data[0])
	print(train_target[0:20])
	print ("The train data has",train_data.shape)
	print ("The  target data has",train_target.shape)
	
	
	test_idx  = np.arange(1001)        
	Train_target = np.delete(train_target,test_idx) #target has next 10,000
	Train_data   = np.delete(train_data,test_idx,axis=0) #data has next 10,000
	
	Test_data=train_data[0:1000] #test_target 
	Test_target=train_target[0:1000]
	
	
	
	
	
	reg = linear_model.LinearRegression()
	
	#from sklearn.linear_model import Lasso
	#reg = Lasso(alpha=0.0001,precompute=True,max_iter=1000,positive=True, random_state=9999, selection='random')
	reg.fit(Train_data,Train_target)
	z = reg.score(Train_data,Train_target)
	print(z)
	"""
	from sklearn.svm import SVR
	reg = SVR(kernel='linear',C=1)
	reg.fit(Train_data,Train_target)
	"""
	
	
	
	prediction = reg.predict(Test_data)
	print(reg.coef_)
	print(prediction[0:20])
	#print(np.mean((prediction-Test_target)**2))
	from sklearn.metrics import r2_score
	print(r2_score(Test_target,prediction))
	
	
		
	"""
	import xgboost as xgb
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	
	#native xgb
	params = {
	    # Parameters that we are going to tune.
	    'max_depth':7,
	    'min_child_weight': 1,
	    'eta':.3,
	    'subsample': 1,
	    'colsample_bytree': 1,
	    # Other parameters
	    'objective':'reg:linear',
	}
	
	params['eval_metric'] = "mae"
	num_boost_round = 999
	model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")],
	    early_stopping_rounds=10
	)
	
	print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))
	
	cv_results = xgb.cv(params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    seed=42,
	    nfold=5,
	    metrics={'mae'},
	    early_stopping_rounds=10
	)
	print(cv_results)
	print(cv_results['test-mae-mean'].min())
	
	#max_depth , min_child_weight
	# You can try wider intervals with a larger step between
	# each value and then narrow it down. Here after several
	# iteration I found that the optimal value was in the
	# following ranges.
	gridsearch_params = [
	    (max_depth, min_child_weight)
	    for max_depth in range(5,12)
	    for min_child_weight in range(1,8)
	]
	
	min_mae = float("Inf")
	best_params = None
	for max_depth, min_child_weight in gridsearch_params:
	    print("CV with max_depth={}, min_child_weight={}".format(
		                     max_depth,
		                     min_child_weight))

	    # Update our parameters
	    params['max_depth'] = max_depth
	    params['min_child_weight'] = min_child_weight
	
	cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    	)
        
         # Update best MAE
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = (max_depth,min_child_weight)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
	
	params['max_depth'] = 11
	params['min_child_weight'] = 7
	
	gridsearch_params = [
	    (subsample, colsample)
	    for subsample in [i/10. for i in range(7,15)]
	    for colsample in [i/10. for i in range(7,15)]
	]
	
	min_mae = float("Inf")
	best_params = None
	
	for subsample, colsample in reversed(gridsearch_params):
		print("CV with subsample={}, colsample={}".format(subsample,colsample))
		params['subsample'] = subsample
		params['colsample_bytree'] = colsample
	
	cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
        )
        # Update best score
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = (subsample,colsample)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
	
	params['subsample'] = .7
	params['colsample_bytree'] = .7
	
	min_mae = float("Inf")
	best_params = None
	
	for eta in [.3, .2, .1, .05, .01, .005]:
    		print("CV with eta={}".format(eta))
    		
    		params['eta'] = eta
	cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
          
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = eta

	print("Best params: {}, MAE: {}".format(best_params, min_mae))
	
	print(params)
	
	model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")],
	    early_stopping_rounds=10
	)
	print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
	
	num_boost_round = model.best_iteration + 1

	best_model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")]
	)
	best_model.save_model("my_model.model")
	loaded_model = xgb.Booster()
	loaded_model.load_model("my_model.model")
	loaded_model.predict(dtest)
	
	"""
		
	"""
	import xgboost as xgb
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	
	#native xgb
	params = {
	    # Parameters that we are going to tune.
	    'max_depth':7,
	    'min_child_weight': 1,
	    'eta':.3,
	    'subsample': 1,
	    'colsample_bytree': 1,
	    # Other parameters
	    'objective':'reg:linear',
	}
	
	params['eval_metric'] = "mae"
	num_boost_round = 999
	model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")],
	    early_stopping_rounds=10
	)
	
	print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))
	
	cv_results = xgb.cv(params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    seed=42,
	    nfold=5,
	    metrics={'mae'},
	    early_stopping_rounds=10
	)
	print(cv_results)
	print(cv_results['test-mae-mean'].min())
	
	#max_depth , min_child_weight
	# You can try wider intervals with a larger step between
	# each value and then narrow it down. Here after several
	# iteration I found that the optimal value was in the
	# following ranges.
	gridsearch_params = [
	    (max_depth, min_child_weight)
	    for max_depth in range(5,12)
	    for min_child_weight in range(1,8)
	]
	
	min_mae = float("Inf")
	best_params = None
	for max_depth, min_child_weight in gridsearch_params:
	    print("CV with max_depth={}, min_child_weight={}".format(
		                     max_depth,
		                     min_child_weight))

	    # Update our parameters
	    params['max_depth'] = max_depth
	    params['min_child_weight'] = min_child_weight
	
	cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    	)
        
         # Update best MAE
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = (max_depth,min_child_weight)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
	
	params['max_depth'] = 11
	params['min_child_weight'] = 7
	
	gridsearch_params = [
	    (subsample, colsample)
	    for subsample in [i/10. for i in range(7,15)]
	    for colsample in [i/10. for i in range(7,15)]
	]
	
	min_mae = float("Inf")
	best_params = None
	
	for subsample, colsample in reversed(gridsearch_params):
		print("CV with subsample={}, colsample={}".format(subsample,colsample))
		params['subsample'] = subsample
		params['colsample_bytree'] = colsample
	
	cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
        )
        # Update best score
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = (subsample,colsample)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
	
	params['subsample'] = .7
	params['colsample_bytree'] = .7
	
	min_mae = float("Inf")
	best_params = None
	
	for eta in [.3, .2, .1, .05, .01, .005]:
    		print("CV with eta={}".format(eta))
    		
    		params['eta'] = eta
	cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
          
	mean_mae = cv_results['test-mae-mean'].min()
	boost_rounds = cv_results['test-mae-mean'].argmin()
	print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	if mean_mae < min_mae:
		min_mae = mean_mae
		best_params = eta

	print("Best params: {}, MAE: {}".format(best_params, min_mae))
	
	print(params)
	
	model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")],
	    early_stopping_rounds=10
	)
	print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
	
	num_boost_round = model.best_iteration + 1

	best_model = xgb.train(
	    params,
	    dtrain,
	    num_boost_round=num_boost_round,
	    evals=[(dtest, "Test")]
	)
	best_model.save_model("my_model.model")
	loaded_model = xgb.Booster()
	loaded_model.load_model("my_model.model")
	loaded_model.predict(dtest)
	
	"""
	
		
