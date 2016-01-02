import numpy as np
import pandas as pd
from sklearn import grid_search, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
import numpy as np
import xgboost as xgb
from scipy import sparse

indices = np.fromfile('train.indices', np.int32)
data = np.fromfile('train.data', np.float32)
indptr = np.fromfile('train.indptr', np.int32)
y = np.fromfile('train.y', np.int32)

indices_test = np.fromfile('test.indices', np.int32)
data_test = np.fromfile('test.data', np.float32)
indptr_test = np.fromfile('test.indptr', np.int32)

print (indices.shape, data.shape, indptr.shape, y.shape)
print (indices_test.shape, data_test.shape, indptr_test.shape)

sp = sparse.csr_matrix((data, indices, indptr), dtype=np.float32)
sp_test = sparse.csr_matrix((data_test, indices_test, indptr_test))
print (sp.shape, sp_test.shape, y.shape)

print (np.histogram(y, 2))
print (y)

params = {'booster':'gbtree',
     'max_depth':7,
#     'min_child_weight':4,
     'eta':0.1,
#     'gamma':0.25,
     'silent':1,
     'objective':'binary:logistic',
#     'lambda':1.,
#     'alpha':1.0,
#      'lambda_bias':0.5,
     'nthread':8,
#      'max_delta_step': 1,
     'subsample':0.83,
#     'num_class':2,
      'colsample_bytree':0.77,
     'eval_metric':'auc'
     }
DX = xgb.DMatrix(sp, y)

#xgb.cv(params=params, dtrain=DX, nfold=2, show_progress=True, num_boost_round=1800)

#qwe
#clf = xgb.sklearn.XGBClassifier(max_depth=9, objective='multi:softprob', n_estimators=1100, 
#                                learning_rate=0.03, subsample=0.9, colsample_bytree=0.9)

train_x, val_x, train_y, val_y = cross_validation.train_test_split(sp, y, test_size=0.2, random_state=124)

DX = xgb.DMatrix(train_x, label=train_y)
VX = xgb.DMatrix(val_x)

bst = xgb.Booster(params, [DX, VX])
for i in range(200):
    bst.update(DX, i)
    preds = bst.predict(VX)
    print ('iteration: ', i, metrics.roc_auc_score(val_y, preds))

#clf.fit(sp, y, eval_metric='mlogloss')
#preds = clf.predict_proba(sp_test)
DT = xgb.DMatrix(sp_test)
preds = bst.predict(DT)

print (preds.shape)
sub = pd.read_csv('sample_submission.csv')
sub.ix[:, 1:] = preds
sub.to_csv('res7.csv', index=False, float_format='%.5f')


 