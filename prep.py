import pandas as pd
import numpy as np
import csv

#train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


sam = pd.read_csv('./sample_submission.csv')

x1 = sam[sam.columns[0]].values
print (x1)
x2 = test[test.columns[0]].values
#print (x2.shape)
#print (np.sum(x1 == x2))


#f1 = open('types', 'w')
#nm = list(test.columns.values)
#for a, b in zip(nm, test.dtypes):
#    print(a, b, file=f1)

#train.to_csv('prep_train.csv', sep=';', index=None, quoting=csv.QUOTE_NONNUMERIC)
#test.to_csv('prep_test.csv', sep=';', index=None, quoting=csv.QUOTE_NONNUMERIC)

