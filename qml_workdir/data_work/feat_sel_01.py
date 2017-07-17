import datetime
import numpy as np

from hyperopt import hp, fmin, tpe

import qml_workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg
from qml_workdir.classes.models import qm



cv = QCV(qm)

model_id = qm.add_by_params(
    QXgb(
** {"alpha": 0.008, "booster": "gbtree", "colsample_bylevel": 0.9, "colsample_bytree": 0.9, "eta": 0.0024, "eval_metric": "logloss",
    "gamma": 0.04, "max_depth": 4, "num_boost_round": 2619, "objective": "binary:logistic", "subsample": 0.7, "tree_method": "hist"}
    ),
    'hyperopt xgb'
)



cv.features_sel_del(model_id, 62, early_stop_cv=lambda x: x>0.545, log_file='qml_workdir/logs/feat16.txt', exclude=['height'])
