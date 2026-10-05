
#------------------Z Separation ML Model-----------------------


#----Imports
import seaborn as sns
import warnings
import os
import joblib
import pandas as pd
import numpy as np
from copy import deepcopy
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict,\
    train_test_split, cross_validate, StratifiedKFold, GridSearchCV,\
    RandomizedSearchCV
from sklearn.feature_selection import RFECV, VarianceThreshold


#----Load Data
def load_df_featurized(csv_path):
    # read lowest_energy from pickle to dataframe
    return pd.read_pickle(csv_path)

save_data = ''
df_ML = load_df_featurized('featurized_Janus_Bulk_dataframe.pickle')

from copy import deepcopy
input_data_zsep = deepcopy(df_ML)
names = input_data_zsep['name'].values
targets = input_data_zsep['z_sep'].values

#-----Reduce to Selected Features
feat_included = ['e_form', 'c2db_form', '2d-y_atomic_number', 'surf_energy',
 'sub_atomic_number', 'radii_sub', '2d max packing efficiency',  # 'delta_2d_z',
 'sub max packing efficiency', 'PymatgenData range molar_volume',
 'avg s valence electrons']

for col in input_data_zsep.columns.values:
    if col not in feat_included:
        input_data_zsep = input_data_zsep.drop(columns = col)

print('The starting number of features is',len(input_data_zsep.columns))


#------Set Model Parameters
bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators=False, 50, 'sqrt', 1, 2, 500
n_splits = 5
random_state = 10

#-----Define Random Forest Regression Model
rf = RandomForestRegressor(n_estimators=n_estimators,
                           random_state=random_state,
                           max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, 
                           min_samples_split=min_samples_split,
                           max_features=max_features,
                           bootstrap=bootstrap)
#------Fit Model

rf.fit(input_data_zsep, targets)
# cross validate to ensure no overfitting
crossvalidation = KFold(n_splits=n_splits,
                        shuffle=True,
                        random_state=random_state)


#-------Extract Statistics
scores = cross_val_score(rf, input_data_zsep, targets, 
                         scoring='neg_mean_squared_error', 
                         cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, input_data_zsep, targets, scoring='r2', 
                            cv=crossvalidation, n_jobs=-1)
mae = cross_val_score(rf, input_data_zsep, targets, 
                         scoring='neg_mean_absolute_error', 
                         cv=crossvalidation, n_jobs=-1)
print('R2 Scores:  ', r2_scores)
print('MSE Scores: ', scores)
print('RMSE Scores:', rmse_scores)
print('MAE Scores: ', mae)
print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
print('Folds: %i, mean MAE = %.3f' % (len(scores), np.mean(np.abs(mae))))


#------Extract Most Important Features and Predicted z-Separations
importances = rf.feature_importances_

predicted = cross_val_predict(rf, input_data_zsep, targets,
            cv=crossvalidation)

