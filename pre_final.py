# I decided to convert Jupyter notebook to ordinary python script for straightforward evaluation

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lightgbm.sklearn import LGBMClassifier

import scoring # local file



with open('type_converter_dict.json', 'r') as f:
    tcd = f.read()
    type_converter = json.loads(tcd)
    revert_to_type = {
        'np.int32' : np.int32,
        'np.float32' : np.float32,
        'object' : object,
    }
    for key in type_converter:
        type_converter[key] = revert_to_type[type_converter[key]]

columns = [key for key in type_converter if not key.startswith('FOI_hits_') or key == 'FOI_hits_N']

data1 = pd.read_csv('../data/train_part_1_v2.csv.gz', dtype=type_converter, usecols=columns, index_col='id')
data2 = pd.read_csv('../data/train_part_2_v2.csv.gz', dtype=type_converter, usecols=columns, index_col='id')
data = pd.concat((data1, data2), axis=0, ignore_index=True)
data1, data2 = 0, 0


test_columns = list(set(columns) - (set(columns) & set(['weight', 'label', 'sWeight', 'kinWeight', 'particle_type'])))
test_final = pd.read_csv('../data/test_private_v3_track_1.csv.gz', usecols=test_columns, index_col='id')



# preprocessing functions
def add_null_feature(data):
    data['miss_2'] = (data['MatchedHit_X[2]'] == -9999).astype(int)
    data['miss_3'] = (data['MatchedHit_X[3]'] == -9999).astype(int)
    return data

def set_null_xy(data):
    idx = data['MatchedHit_X[2]'] == -9999
    data.loc[idx, 'MatchedHit_X[2]'] = data[idx]['MatchedHit_X[1]'] - data[idx]['MatchedHit_X[0]']
    data.loc[idx, 'MatchedHit_X[3]'] = data[idx]['MatchedHit_X[2]'] - data[idx]['MatchedHit_X[1]']
    data.loc[idx, 'MatchedHit_Y[2]'] = data[idx]['MatchedHit_X[1]'] - data[idx]['MatchedHit_X[0]']
    data.loc[idx, 'MatchedHit_Y[3]'] = data[idx]['MatchedHit_X[2]'] - data[idx]['MatchedHit_X[1]']
    return data

def set_uncrossed_time(data):
    for i in range(4):
        idx = (data['MatchedHit_TYPE[%i]' % i] == 1) | (data['MatchedHit_TYPE[%i]' % i] == 0)
        data.loc[idx, 'MatchedHit_DT[%i]' % i] = 15
        
    return data

def add_growing_time(data):
    d = data
    mht = 'MatchedHit_T'
    idx = (d[mht+'[0]'] <= d[mht+'[1]']) & (d[mht+'[1]'] <= d[mht+'[2]']) & (d[mht+'[2]'] <= d[mht+'[3]']) & (d[mht+'[3]'] != 255)
    data['time_grows'] = idx.astype(int)
    return data

def add_sum_dtime(data):
    mht = 'MatchedHit_DT'
    data['sum_dtime'] = data[mht+'[0]'] + data[mht+'[1]'] + data[mht+'[2]'] + data[mht+'[3]']
    return data

def add_diff_xy(data):
    x = data
    med_x = np.abs(np.median([x['MatchedHit_X[1]'] - x['MatchedHit_X[0]'], 
                              x['MatchedHit_X[2]'] - x['MatchedHit_X[1]'], 
                              x['MatchedHit_X[3]'] - x['MatchedHit_X[2]']], axis=0))
    diff_x = [np.abs(np.abs(x['MatchedHit_X[1]'] - x['MatchedHit_X[0]']) - med_x) - (x['MatchedHit_DX[1]']+x['MatchedHit_DX[0]'])/4,
              np.abs(np.abs(x['MatchedHit_X[2]'] - x['MatchedHit_X[1]']) - med_x) - (x['MatchedHit_DX[2]']+x['MatchedHit_DX[1]'])/4,
              np.abs(np.abs(x['MatchedHit_X[3]'] - x['MatchedHit_X[2]']) - med_x) - (x['MatchedHit_DX[3]']+x['MatchedHit_DX[2]'])/4]
    diff_x = np.clip(diff_x, a_min=0, a_max=9000)
    divergence_x = np.sum(np.square(diff_x), axis=0)


    med_y = np.abs(np.median([x['MatchedHit_Y[1]'] - x['MatchedHit_Y[0]'], 
                              x['MatchedHit_Y[2]'] - x['MatchedHit_Y[1]'], 
                              x['MatchedHit_Y[3]'] - x['MatchedHit_Y[2]']], axis=0))
    diff_y = [np.abs(np.abs(x['MatchedHit_Y[1]'] - x['MatchedHit_Y[0]']) - med_y) - (x['MatchedHit_DY[1]']+x['MatchedHit_DY[0]'])/4,
              np.abs(np.abs(x['MatchedHit_Y[2]'] - x['MatchedHit_Y[1]']) - med_y) - (x['MatchedHit_DY[2]']+x['MatchedHit_DY[1]'])/4,
              np.abs(np.abs(x['MatchedHit_Y[3]'] - x['MatchedHit_Y[2]']) - med_y) - (x['MatchedHit_DY[3]']+x['MatchedHit_DY[2]'])/4]
    diff_y = np.clip(diff_y, a_min=0, a_max=9000)
    divergence_y = np.sum(np.square(diff_y), axis=0)
    return pd.concat((data, 
                      pd.Series(divergence_x, name='div_x'), 
                      pd.Series(divergence_y, name='div_y')), axis=1)


def add_Lextra_dirs(data):
    dir_x = data['Lextra_X[3]'] - data['Lextra_X[0]']
    dir_y = data['Lextra_Y[3]'] - data['Lextra_Y[0]']
    return pd.concat((data, 
                      pd.Series(dir_x, name='lextra_dir_x'), 
                      pd.Series(dir_y, name='lextra_dir_y')), axis=1)

def add_MatchedHit_dirs(data):
    dir_x = np.median([data['MatchedHit_X[1]'] - data['MatchedHit_X[0]'], 
                       data['MatchedHit_X[2]'] - data['MatchedHit_X[1]'], 
                       data['MatchedHit_X[3]'] - data['MatchedHit_X[2]']], axis=0)
    dir_y = np.median([data['MatchedHit_Y[1]'] - data['MatchedHit_Y[0]'], 
                       data['MatchedHit_Y[2]'] - data['MatchedHit_Y[1]'], 
                       data['MatchedHit_Y[3]'] - data['MatchedHit_Y[2]']], axis=0)
    return pd.concat((data, 
                      pd.Series(dir_x, name='matched_dir_x'), 
                      pd.Series(dir_y, name='matched_dir_y')), axis=1)

def add_Lextra_Matched_diffs_relative(data):
    diff_x_rel = np.abs((data['lextra_dir_x'] - data['matched_dir_x']) / data['matched_dir_x'])
    diff_y_rel = np.abs((data['lextra_dir_y'] - data['matched_dir_y']) / data['matched_dir_y'])
    return pd.concat((data, 
                      pd.Series(diff_x_rel, name='LM_diff_x_rel'), 
                      pd.Series(diff_y_rel, name='LM_diff_y_rel')), axis=1)


def drop_MatchedHit_DZ_FEATURES(data):
    return data.drop(['MatchedHit_DZ[0]', 'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]'], axis=1)


def basic_preprocess(data):
    data = add_null_feature(data)
    data = set_null_xy(data)
    data = set_uncrossed_time(data)
    data = add_growing_time(data)
    data = add_sum_dtime(data)
    data = add_diff_xy(data)
    data = add_Lextra_dirs(data)
    data = add_MatchedHit_dirs(data)
    data = add_Lextra_Matched_diffs_relative(data)
    data = drop_MatchedHit_DZ_FEATURES(data)
    
    data['ndof'] = data['ndof'].map({4:0, 6:1, 8:2})
    data['NShared'] = data['NShared'].apply(lambda x: x if x < 3 else 2)
    return data

def train_preprocess(data):
    data = basic_preprocess(data)
    
    return data
    

def test_preprocess(data):
    data = basic_preprocess(data)
    return data

def get_data_train(data):
    return data.drop(['kinWeight', 'label', 'particle_type', 'sWeight', 'weight'], axis=1)


scaling_columns = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]', 'avg_cs[1]',
                   'avg_cs[2]', 'avg_cs[3]', 
                   'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
                   'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
                   'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
                   'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
                   'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
                   'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
                   'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_T[0]',
                   'MatchedHit_T[1]', 'MatchedHit_T[2]', 'MatchedHit_T[3]',
                   'MatchedHit_DT[0]', 'MatchedHit_DT[1]', 'MatchedHit_DT[2]',
                   'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]',
                   'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]',
                   'Lextra_Y[3]', 'Mextra_DX2[0]', 'Mextra_DX2[1]',
                   'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]', 'Mextra_DY2[1]',
                   'Mextra_DY2[2]', 'Mextra_DY2[3]', 'PT', 'P', 
                   'sum_dtime', 'div_x', 'div_y', 'lextra_dir_x',
                   'lextra_dir_y', 'matched_dir_x', 'matched_dir_y', 'LM_diff_x_rel',
                   'LM_diff_y_rel']

stdscalers = {col:StandardScaler() for col in scaling_columns}

def scale_fit(data):
    for col in scaling_columns:
        stdscalers[col].fit(data[col].values.reshape(-1, 1))
    return data

        
def scale_transform(data):
    for col in scaling_columns:
        data[col] = stdscalers[col].transform(data[col].values.reshape(-1, 1))
    return data


def train_kfold_lgbms(X, y, weights, n_folds=3):
    lgbms = [LGBMClassifier(n_estimators=800, 
                            learning_rate=0.2, 
                            min_data_in_leaf=200, 
                            max_depth=8, 
                            num_leaves=80, 
                            max_bin=256,
                            importance_type='gain')
             for i in range(n_folds)]
    
    skf = StratifiedKFold(n_splits=n_folds)
    i = 0
    for train_index, valid_index in skf.split(X, y):
        
        swegh = np.abs(weights.loc[X.iloc[train_index].index.values])
        swegh[swegh>100] = 100
        
        lgbms[i].fit(X.iloc[train_index], y.iloc[train_index], 
                     eval_set=(X.iloc[valid_index], y.iloc[valid_index]), 
                     early_stopping_rounds=10, 
                     categorical_feature=['ndof', 'NShared', 'miss_2', 'miss_3', 'time_grows',
                                          'MatchedHit_TYPE[0]','MatchedHit_TYPE[1]',
                                          'MatchedHit_TYPE[2]','MatchedHit_TYPE[3]',],
                     verbose=10,
                     sample_weight=swegh) # regretfully, i forgot to fix random seed, so result is unrepeatable :(
        print('trained {0}/{1}'.format(i+1, n_folds))
        i += 1
        
    return lgbms

def get_kfold_prediction(models, X_test):
    predictions = np.ndarray((len(models), X_test.shape[0]))
    for i, model in enumerate(models):
        predictions[i] = model.predict_proba(X_test)[:,1]
        
    return predictions


# train
data = train_preprocess(data)
data_train = get_data_train(data)
data = data[['kinWeight', 'label', 'particle_type', 'sWeight', 'weight']]

scale_fit(data_train)
data_train = scale_transform(data_train)

lgbms = train_kfold_lgbms(data_train, data['label'], data['weight'], 4)


# test
test_final = test_preprocess(test_final)
test_final = scale_transform(test_final)

predictions_lgbm = get_kfold_prediction(models=lgbms, X_test=test_final)
pred_lgbm = predictions_lgbm.mean(axis=0)

pd.DataFrame(data={"prediction": pred_lgbm}, index=test_final.index).to_csv("../pre_final_submission.csv", index_label='id')

