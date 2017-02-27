from __future__ import print_function, division
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.model_selection import GridSearchCV

#features to remove
catvar = ['recid','sc115', 'sc116' ,'uf30',	'uf29',	'rec8',	'rec7', 'fw', 'chufw', 'seqno', 'flg_sx1',	'flg_ag1', 'flg_hs1', 'flg_rc1', 'hflag2', 'hflag1', 'hflag13',	'hflag6', 'hflag3',	
		'hflag14', 'hflag16', 'hflag7', 'hflag9', 'hflag10', 'hflag91',	'hflag11',	'hflag12',	'hflag4', 'hflag18',	'uf52h_h',	'uf52h_a',	'uf52h_b',	'uf52h_c',	'uf52h_d', 'uf52h_e',
		'uf52h_f', 'uf52h_g', 'rec53',	'tot_per',	'rec28', 'uf26', 'uf28', 'uf27', 'rec39', 'uf42', 'uf42a',	'uf34',	'uf34a', 'uf35', 'uf35a', 'uf36', 'uf36a', 'uf37', 'uf37a', 'uf38',
		'uf38a', 'uf39', 'uf39a', 'uf40', 'uf40a', 'sc27', 'rec1',	'uf46',	'rec4',	'rec_race_a',	'rec_race_c','sc26', 'uf53', 'uf54', 'uf19', 'new_csr', 'uf17a', 'sc184',
		'sc159', 'uf12', 'sc161', 'uf13', 'uf14', 'sc164', 'uf15', 'uf2a','uf2b', 'hhr2', 'uf43', 'hhr5', 'race1', 'sc51', 'sc52',	'sc53','sc54', 'sc110', 'sc111', 'sc112', 'sc113','rec21','uf5','uf6'
]

#features to cleanse NaN(999 etc..)
nanlist = ['uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6', 'uf1_7',
	'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13', 'uf1_14', 'uf1_15',
	'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18', 'uf1_19', 'uf1_20', 'uf1_21', 'uf1_22',
	'sc23' ,'sc24', 'sc36', 'sc37', 'sc38','sc117', 'sc118','sc120', 'sc121','sc125','sc140', 'sc141','sc143', 
	'sc144','sc147','sc173', 'sc171','sc154','sc157','sc174', 'sc181', 'sc541',
	'sc542', 'sc543', 'sc544', 'sc185', 'sc186', 'sc197', 'sc198' , 'sc187', 'sc188',
	'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 'sc193', 'sc194', 'sc196', 'sc548',
	'sc549', 'sc550', 'sc551', 'sc199', 'sc575', 'sc574', 'sc560']


def predict_rent():
	'''
		With NYC Housing and Vacancy Survey data, predict the monthly contract rent(UF17).
		Used GridSearchCV with the Lasso model, and tuned epsilon and alphas.

		return: test data, test label, predicted label
	'''

	global nanlist
	df = pd.read_csv("homework2_data.csv")

	df = df[df.uf17 != 99999]    # remove rows of unlabeled response variable
	y = df['uf17']
	del df['uf17']

	for i in catvar:             # remove irrelevant feature columns
		del df[i]

	for i in nanlist:            # replace Not reported data to NaN for Imputation
		df[i].replace(9, np.nan, inplace=True)

	nanlist = ['sc134', 'uf7a','uf8','uf64']   
	for i in nanlist:
		df[i].replace(9999, np.nan, inplace=True)
	
	X_train, X_test, y_train, y_test = train_test_split(df,y, random_state=0)

	imp = Imputer(strategy='most_frequent').fit(X_train)
	X_train = imp.transform(X_train)
	X_test = imp.transform(X_test)

	EN = LassoCV()             #prepare Lasso model and tuning hyperparameters
	param_grid = {'eps' : [0.001, 0.05, 0.1], 'n_alphas':[50, 100, 150]}
	pipe = make_pipeline(StandardScaler(), GridSearchCV(EN, param_grid, cv=10))
	pipe.fit(X_train, y_train)

	return X_test, y_test, pipe.predict(X_test)
	


def score_rent():
	'''
		With NYC Housing and Vacancy Survey data, score predicted labels of the monthly contract rent(UF17) by R^2 metric.
		Used GridSearchCV with the Lasso model, and tuned epsilon and alphas.

		return: R^2 of predicted label
	'''

	global nanlist
	df = pd.read_csv("homework2_data.csv")
	df = df[df.uf17 != 99999]          # remove rows of unlabeled response variable
	y = df['uf17']
	del df['uf17']

	for i in catvar:
		del df[i]

	for i in nanlist:                 # remove irrelevant feature columns
		df[i].replace(9, np.nan, inplace=True)

	nanlist = ['sc134', 'uf7a','uf8','uf64']
	for i in nanlist:
		df[i].replace(9999, np.nan, inplace=True)
	
	X_train, X_test, y_train, y_test = train_test_split(df,y, random_state=0)
	imp = Imputer(strategy='most_frequent').fit(X_train)         # replace Not reported data to NaN for Imputation
	X_train = imp.transform(X_train)
	X_test = imp.transform(X_test)

	EN = LassoCV()                     #prepare Lasso model and tuning hyperparameters
	param_grid = {'eps' : [0.001, 0.05, 0.1], 'n_alphas':[50, 100, 150]}
	pipe = make_pipeline(StandardScaler(), GridSearchCV(EN, param_grid, cv=10))
	pipe.fit(X_train, y_train)
	return r2_score(y_test, pipe.predict(X_test))