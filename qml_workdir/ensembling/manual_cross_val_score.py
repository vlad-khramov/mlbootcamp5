import datetime
import numpy as np
import pandas as pd

import os
import sys

from sklearn import model_selection
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.getcwd())

from  qml_workdir.classes.config import config

from qml.helpers import load

from qml.cv import QCV
from qml.models import QXgb
from qml_workdir.classes.models import qm

cv = QCV(qm)

X = pd.read_csv(config.QML_TRAIN_X_FILE_MASK.format(69), index_col='id')
Y = pd.read_csv(config.QML_TRAIN_Y_FILE_MASK.format(69), index_col='id')
test = pd.read_csv(config.QML_TEST_X_FILE_MASK.format(69), index_col='id')

#
model = qm._load_model(1747)
# splits = load(cv._get_splits_filename(5, Y.index))
# print(splits)
# exit()

predicted = cross_val_predict(model, X, Y, cv=5)

print(predicted)

# np.savetxt('qml_workdir/ensembling/manual02/0', res)
