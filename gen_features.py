import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import scipy
from scipy import stats

seed = 260681

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

y = train.QuoteConversion_Flag.values.astype(np.int32)
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

def count_less_0(df):
    df["Below0"] = np.sum(df<0, axis = 1)
    return df

def count_0(df):
    cols = [col for col in df.columns if col != "QuoteConversion_Flag"]
    df["CountZero"]=np.sum(df[cols] == 0, axis = 1)
    return df

train = count_less_0(train)
test = count_less_0(test)

#train['Year_cat'] = train['Year'].astype('object')
#test['Year_cat'] = test['Year'].astype('object')

train['Month_cat'] = train['Month'].astype('object')
test['Month_cat'] = test['Month'].astype('object')

train['weekday_cat'] = train['weekday'].astype('object')
test['weekday_cat'] = test['weekday'].astype('object')

mask = (train.dtypes=='object').values

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

one = preprocessing.OneHotEncoder(categorical_features=mask, handle_unknown='ignore')
one.fit(test.values)
train = one.transform(train.values)
test = one.transform(test.values)
print (train.shape, test.shape)

train = scipy.sparse.csr_matrix(train, dtype=np.float32)
train.indices.tofile('train.indices')
train.data.tofile('train.data')
train.indptr.tofile('train.indptr')
y.tofile('train.y')

test = scipy.sparse.csr_matrix(test, dtype=np.float32)
test.indices.tofile('test.indices')
test.data.tofile('test.data')
test.indptr.tofile('test.indptr')
