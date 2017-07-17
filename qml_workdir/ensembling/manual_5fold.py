import datetime
import numpy as np
import pandas as pd

import os
import sys

from sklearn import model_selection

sys.path.insert(0, os.getcwd())

from  qml_workdir.classes.config import config
from qml.cv import QCV
from qml.models import QXgb
from qml_workdir.classes.models import qm

cv = QCV(qm)

X = pd.read_csv(config.QML_TRAIN_X_FILE_MASK.format(69), index_col='id')
Y = pd.read_csv(config.QML_TRAIN_Y_FILE_MASK.format(69), index_col='id')
test = pd.read_csv(config.QML_TEST_X_FILE_MASK.format(69), index_col='id')


splits = cv._load_splits(Y)

(train_indexes, test_indexes) = splits[4]

X_train = X.loc[train_indexes]
Y_train = Y.loc[train_indexes][config.QML_RES_COL]
X_test = X.loc[test_indexes]
Y_test = Y.loc[test_indexes][config.QML_RES_COL]

model = qm._load_model(1747)

res = []

# # model.fit(X_test, Y_test, seed=1000)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1001)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1002)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1003)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1004)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1005)
# # res.append(model.predict(test))
# #
# # model.fit(X_test, Y_test, seed=1006)
# # res.append(model.predict(test))
# # model.fit(X_test, Y_test, seed=1007)
# # res.append(model.predict(test))
# #
# # res = np.array(res)
# np.savetxt('qml_workdir/ensembling/manual02/0', res)
#
# np.savetxt('qml_workdir/ensembling/manual02/1', res.mean())
# np.savetxt('qml_workdir/ensembling/manual02/2', res.T.mean())
#
#

r = np.loadtxt('qml_workdir/ensembling/manual02/0')

np.savetxt('qml_workdir/ensembling/manual02/5', np.array([i.mean() for i in r.T]))