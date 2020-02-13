from process_data import *
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle

#make sure GPU is being used 
print(tf.test.is_gpu_available())

data_input_path = 'input/'
data_output_path = ''
data_output_file = 'characteristics.pkl'
model_output_path = 'models/'
n_stocks = 3000
np.random.seed(0)

def format_data_for_models(df_char,split_train=.7,split_val=.6):
	#input: pandas dataframe of stock characteristics (indexed by [permno,date]), fraction for training, fraction for validation
	#output: train, validation, and test splits for X data (characteristics), Y data (stock returns), and date range
	#X shape: (n_stocks*n_dates,n_characteristics), padded with zeros for dates with <n_stocks; Y shape: (n_stocks*n_dates,)
	char_cols = df_char.columns.values
	n_feats = len(char_cols)
	#fill missing with 0 (cross-sectional mean)
	df_char = df_char.reset_index().fillna(0)
	dates = df_char.date.drop_duplicates().sort_values()
	#initialize padded output with zeros
	X = np.zeros((len(dates),n_stocks,n_feats))
	Y = np.zeros((len(dates),n_stocks))
	D = []
	#merge returns onto characteristics
	crsp = get_crsp_m()
	crsp['date'] = pd.to_datetime(crsp['date'])-MonthBegin(1)
	combined = df_char.merge(crsp[['permno','date','retadj']],how='left',on=['permno','date'])
	#for each date, update output with X and Y values
	for i,d in enumerate(dates):
	    D.append(d)
	    temp = combined.loc[combined.date==d]
	    X[i,:temp.shape[0],:temp.shape[1]] = temp[char_cols].values
	    Y[i,:temp.shape[0]] = temp['retadj'].values
	D=np.array(D)
	#holdout test set from end of date range
	n_d_train = int(np.floor(split_train*len(D)))
	X_test,Y_test,D_test = X[n_d_train:],Y[n_d_train:],D[n_d_train:]
	X_trainval,Y_trainval,D_trainval = X[:n_d_train],Y[:n_d_train],D[:n_d_train]
	#get random partitions of earlier data for train and validation sets
	part = np.random.permutation(X_trainval.shape[0])
	n_d_val = int(np.floor(split_val*len(X_trainval)))
	part_train = part[:n_d_val]
	part_val = part[n_d_val:]
	X_train,Y_train,D_train = X_trainval[part_train],Y_trainval[part_train],D_trainval[part_train]
	X_val,Y_val,D_val = X_trainval[part_val],Y_trainval[part_val],D_trainval[part_val]
	#reshape all X and Y data to be indexed by month/stock
	X_train,X_val,X_test = np.reshape(X_train,(X_train.shape[0]*n_stocks,n_feats)),np.reshape(X_val,(X_val.shape[0]*n_stocks,n_feats)),np.reshape(X_test,(X_test.shape[0]*n_stocks,n_feats))
	Y_train,Y_val,Y_test = np.reshape(Y_train,(Y_train.shape[0]*n_stocks,)),np.reshape(Y_val,(Y_val.shape[0]*n_stocks,)),np.reshape(Y_test,(Y_test.shape[0]*n_stocks,))
	data = dict(zip(['X_train','X_val','X_test','Y_train','Y_val','Y_test','D_train','D_val','D_test'],[X_train,X_val,X_test,Y_train,Y_val,Y_test,D_train,D_val,D_test]))
	return data

def loss_sharpe(y_true,y_pred):
	#loss function for models; uses returns (y_true) and predictions (y_pred) to calculate long-short portfolio Sharpe ratio 
	#reshape from long to wide, from (n_months*n_stocks,) to (n_months,n_stocks)
	y_pred = tf.reshape(y_pred,[tf.shape(y_pred)[0]/n_stocks,n_stocks])
	ret = tf.reshape(y_true,[tf.shape(y_true)[0]/n_stocks,n_stocks])
	#normalize weights to be mean zero, and divide by n_stocks to try to control leverage during training
	w = (y_pred-tf.reduce_mean(y_pred,1,keep_dims=True))/n_stocks
	#monthly return vector
	ret_port = tf.reduce_sum(tf.multiply(ret,w),1)
	#negative of Sharpe ratio
	return -tf.reduce_mean(ret_port)/tf.math.reduce_std(ret_port)

def create_model_linear(lr,l1_penalty,input_shape):
	#linear model with L1 regularization and sharpe_loss loss function
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(1,activation='linear',kernel_regularizer=tf.keras.regularizers.l1(l1_penalty),input_shape=input_shape)
	])
	optimizer = tf.keras.optimizers.Adam(lr)
	model.compile(
		loss=loss_sharpe,
		optimizer = optimizer,
	)
	return model

def create_model_nonlinear(lr,n_layers,n_units,dropout,input_shape):
	#nonlinear model with dropout regularization and sharpe_loss loss function
	#architecture is based on repeated dense/dropout layers with the same number of units and dropout rate
	input_layer = [
		tf.keras.layers.Dense(n_units,activation='relu',input_shape=input_shape),
		tf.keras.layers.Dropout(dropout)
	]
	hidden_layer = []
	for i in range(n_layers-1):
		hidden_layer += [
			tf.keras.layers.Dense(n_units,activation='relu'),
			tf.keras.layers.Dropout(dropout)
		]
	mod = input_layer+hidden_layer
	mod += [tf.keras.layers.Dense(1,activation='sigmoid')]
	model = tf.keras.Sequential(mod)
	optimizer = tf.keras.optimizers.Adam(lr)
	model.compile(
		loss=loss_sharpe,
		optimizer = optimizer,
	)
	print(model.summary())
	return model

def evaluate_model_on_data(mod,X_data,Y_data,date_range):
	#calculates portoflio weights and returns given a trained model and input data (either train, validation, or test data)
	#same steps as loss_sharpe
	pred = np.squeeze(mod.predict(X_data))
	pred = np.reshape(pred,(int(len(pred)/n_stocks),n_stocks))
	ret = np.reshape(Y_data,(int(len(Y_data)/n_stocks),n_stocks))
	w = (pred-pred.mean(1,keepdims=True))/n_stocks
	#these steps are to make the long and short sides sum to 1 (not differentiable in loss function)
	w_long = (w*(w>0)).sum(1,keepdims=True)
	w_short = (w*(w<0)).sum(1,keepdims=True)
	#final portfolio weights
	w = np.where(w>0,w/(w_long),w/(-w_short))
	#final portfolio returns
	ret_port = pd.Series(np.sum(w*ret,1),index=date_range)
	return ret_port,w

def train_and_save_model(dataset,output_name,linear=False,lr=.01,n_layers=1,n_units=16,dropout=.5,l1_penalty=0):
	#creates/trains model based on hyperparameters and saves model and analytics from evaluate_model_on_data 
	#input: dictionary of training and validation data (see format_data_for_models), name to save model as, and hyperparameters 
	#checks whether this model has been output already
	if not os.path.isfile(model_output_path+'model_'+str(output_name)+'.h5'):
		#create model
		input_shape = dataset['X_train'][0].shape
		if linear:
			model = create_model_linear(lr,l1_penalty,input_shape)
		else:
			model = create_model_nonlinear(lr,n_layers,n_units,dropout,input_shape)
		#fit model on data; note that epochs and batch_size are fixed but could be hyperparameters
		#batch_size is set to 10 years of data at a time to avoid memory errors 
		history = model.fit(dataset['X_train'],dataset['Y_train'],epochs=60,batch_size=n_stocks*120,shuffle=False,validation_data=(dataset['X_val'],dataset['Y_val']))
		model.save(model_output_path+'model_'+str(output_name)+'.h5')
		with open(model_output_path+'history_'+str(output_name)+'.p','wb') as f:
			pickle.dump(history.history,f)
		#initialize dictionary of portfolio returns from trained models
		if os.path.isfile(model_output_path+'model_returns.p'):
			with open(model_output_path+'model_returns.p','rb') as f:
				model_returns = pickle.load(f)
				model_returns[output_name] = {}
		else:
			model_returns={output_name:{}}
		#initialize dictionary of portfolio weights from trained models
		if os.path.isfile(model_output_path+'model_weights.p'):	
			with open(model_output_path+'model_weights.p','rb') as f:
				model_weights = pickle.load(f)
				model_weights[output_name] = {}
		else:
			model_weights={output_name:{}}
		#get portfolio returns and weights from new trained model by evaluating on train, validation, and test data 
		for j in ['train','val','test']:
			ret,w = evaluate_model_on_data(model,dataset['X_'+j],dataset['Y_'+j],dataset['D_'+j])
			model_returns[output_name][j]=ret
			model_weights[output_name][j]=w
		#overwrite dictionaries with new data
		with open(model_output_path+'model_returns.p','wb') as f:
			pickle.dump(model_returns,f)
		with open(model_output_path+'model_weights.p','wb') as f:
			pickle.dump(model_weights,f)

def get_model_specs(n_linear=1,n_nonlinear=1):
	#generate a dictionary of model hyperparameter specifications; n_linear/n_nonlinear give the number of initializations to generate for each specification
	#if already output, read in the pickled file
	if os.path.isfile(model_output_path+'model_specifications.p'):
		with open(model_output_path+'model_specifications.p','rb') as f:
			model_specs = pickle.load(f)
	#if not already output, generate model specifications based on grid of hyperparameters (or randomly generated ones)
	else:
		model_params = []
		for l in [1,2,3]:
			for u in [16,32,64]:
				for d in [.25,.5,.75]:
					for _ in range(n_nonlinear):
						#set to .01 but could be randomly generated a number of times
						rate = .01
						#rate = 10**(-2*np.random.rand()-1)
						model_params.append({'linear':False,'lr':rate,'n_layers':l,'n_units':u,'dropout':d})
		for _ in range(n_linear):
			#could be randomly generated a number of times
			rate = .01
			penalty = 0
			#rate = 10**(-2*np.random.rand()-1)
			#penalty = 10**(-3*np.random.rand()-3)
			model_params.append({'linear':True,'lr':rate,'l1_penalty':penalty})
		model_specs = {i:spec for i,spec in enumerate(model_params)}
		#save dictionary as pickle
		with open(model_output_path+'model_specifications.p','wb') as f:
			pickle.dump(model_specs,f)
	return model_specs

if __name__=='__main__':
	characteristics = get_characteristics()
	dataset = format_data_for_models(characteristics)
	model_specs = get_model_specs()
	for i,spec in model_specs.items():
		print('model '+str(i)+' out of '+str(len(model_specs)-1)+' training')
		print('hyperparameters:')
		print(spec)
		train_and_save_model(dataset,i,**spec)
		K.clear_session()
	