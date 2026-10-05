#------------------Binding Energy ML Model-----------------------



#----Imports
import seaborn as sns
import warnings
import os
import numpy as np
import pandas as pd
import joblib
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


#-----Load Data
def load_df_featurized(csv_path):
   
    return pd.read_pickle(csv_path)


save_data = ''
df_ML = load_df_featurized('featurized_Janus_Bulk_dataframe.pickle')


from copy import deepcopy
input_data = deepcopy(df_ML)
names = input_data['name'].values
targets = input_data['e_bind'].values

#-----Reduce to Selected Features
feat_included = ['surf_energy', 'e_form', 'c2db_form', 'c2db_ehull',
       '2d-m_atomic_number', '2d-y_atomic_number', 'sub_atomic_number',
       'radii_2d-m', 'avg_e_2d', '2d max packing efficiency',
       'sub max packing efficiency', 'PymatgenData range molar_volume',
                'avg_e_diff',   
       'n_symmetry_ops',  'avg d valence electrons', 'avg_e_sub']

for col in input_data.columns.values:
    #print(col)
    if col not in feat_included:
        input_data = input_data.drop(columns = col)

print('The starting number of features is',len(input_data.columns))

#import matplotlib
#------Set Model Parameters
n_estimators,n_splits,max_depth,max_features,bootstrap=150,20,50,'sqrt',False
random_state = 10



#-----Define Random Forest Regression Model
rf = RandomForestRegressor(n_estimators=n_estimators,
                           random_state=random_state,
                           max_depth=max_depth,
                           max_features=max_features,
                           bootstrap=bootstrap)

#-----Fit Model

rf.fit(input_data, targets)

crossvalidation = KFold(n_splits=n_splits,
                        shuffle=True,
                        random_state=random_state)


#-------Extract Statistics

scores = cross_val_score(rf, input_data, targets, 
                         scoring='neg_mean_squared_error', 
                         cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, input_data, targets, scoring='r2', 
                            cv=crossvalidation, n_jobs=-1)
mae = cross_val_score(rf, input_data, targets, 
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


#------Extract Most Important Features and Predicted Binding Energies
importances = rf.feature_importances_


predicted = cross_val_predict(rf, input_data, targets,
            cv=crossvalidation)
