import os
import sys



sys.path.insert(0, os.getcwd())


import qml_workdir.classes.config
from qml.cv import QCV
from qml_workdir.classes.models import qm

from qml.models import QXgb, QAvg

cv = QCV(qm)

# Кроссвалидация простой модели на первой версии данных
# print(cv.cross_val(1, 1))

# Предсказание
# qm.qpredict(1, 1)

################################################################################
# Усреднение по 8 сидам

model_id = qm.add_by_params(
    QXgb(
        **{"alpha": 0.008, "booster": "gbtree", "colsample_bylevel": 0.6, "colsample_bytree": 1.0,
           "eta": 0.004, "eval_metric": "logloss", "gamma": 0.2, "max_depth": 4, "num_boost_round": 2015,
           "objective": "binary:logistic", "subsample": 0.7, "tree_method": "hist"}
    ),
    'hyperopt xgb'
)
model_id_avg = qm.add_by_params(
    QAvg(
        [
            [model_id, 69, 1000],
            [model_id, 69, 1001],
            [model_id, 69, 1002],
            [model_id, 69, 1003],
            [model_id, 69, 1004],
            [model_id, 69, 1005],
            [model_id, 69, 1006],
            [model_id, 69, 1007],
        ]

    ), level=-2
)

print(cv.cross_val(model_id_avg, -1))
qm.qpredict(model_id_avg,-1)

