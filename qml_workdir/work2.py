import datetime
import numpy as np

import os
import sys

from sklearn.linear_model import Ridge, RidgeClassifier

sys.path.insert(0, os.getcwd())

from hyperopt import hp, fmin, tpe

import qml_workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg, QRankedByLineAvg, QNN1, QStackModel
from qml_workdir.classes.models import qm



cv = QCV(qm)
#
# cv.features_imp(1129, 3, early_stop_cv=lambda x: x>0.65)
#qm.qpredict(107924,-1)


# model_id = qm.add_by_params(
#     QAvg(
#         [
#             [1395, 66, 1000],
#             [1395, 66, 1001],
#             [1395, 66, 1002],
#             [1395, 66, 1003],
#             [1395, 66, 1004],
#             [1395, 66, 1005],
#             [1395, 66, 1006],
#             [1395, 66, 1007],
#         ]
#
#     ), level=-2
# )
#
# qm.qpredict(model_id,-1)
# print(model_id)
# # #
#
# model_id = qm.add_by_params(
#     QAvg(
#         [
#             [1747, 69, 1000],
#             [1747, 69, 1001],
#             [1747, 69, 1002],
#             [1747, 69, 1003],
#             [1747, 69, 1004],
#             [1747, 69, 1005],
#             [1747, 69, 1006],
#             [1747, 69, 1007],
#         ]
#
#     ), level=-2
# )
#
# qm.qpredict(model_id,-1)
# print(model_id)


#print(model_id)


# model_id = qm.add_by_params(
#     QAvg(
#         [
#             [1395, 52, 1000],
#             [1395, 52, 1001],
#             [1395, 52, 1002],
#             [1395, 52, 1003],
#             [1395, 52, 1004],
#             [1395, 52, 1005],
#             [1395, 52, 1006],
#             [1395, 52, 1007],
#         ]
#
#     )
# )
#
# qm.qpredict(model_id,-1)
# print(model_id)

# qm.qpredict(1175,5)
#
# for data_id in [7]:
#     for model_id in [1131]:
#         res = cv.cross_val(model_id, data_id, force=True,seed=1000)
#         print ('--')
#






#                 #
# model_id = qm.add_by_params(
#     QAvg(
#         [
#             [1747, 69, 1000],
#             [1747, 69, 1001],
#             [1747, 69, 1002],
#             [1747, 69, 1003],
#             [1747, 69, 1004],
#             [1747, 69, 1005],
#             [1747, 69, 1006],
#             [1747, 69, 1007],
#             # [1747, 69, 1008],
#             # [1747, 69, 1009],
#             # [1747, 69, 1010],
#             # [1747, 69, 1011],
#             # [1747, 69, 1012],
#             # [1747, 69, 1013],
#             # [1747, 69, 1014],
#             # [1747, 69, 1015],
#         ]
#
#     ), level=-2
# )
#
# res = cv.cross_val(model_id, -1)
# print(model_id, res)


# if __name__ == '__main__':
#     model_id = qm.add_by_params(
#         Ridge(),
#     )
# 
# 
#     model_id = qm.add_by_params(
#         QStackModel(
#             [
#                 [1747, 69, 1000],
#                 [1747, 66, 1000],
#                 [1747, 47, 1000],
#                 #[1747, 69, 1001],
#                  [101340, 69, 1000],
#                  [101340, 66, 1000],
#                  [101340, 47, 1000],
#                 #[101340, 69, 1001],
#                 [101331, 69, 1000],
#                 [101331, 66, 1000],
#                 [101331, 47, 1000],
#                 #[101331, 69, 1001],
#                 [101261, 69, 1000],
#                 [101261, 66, 1000],
#                 [101261, 47, 1000],
# 
#                 [101655, 266, 1000],
#                 [101655, 269, 1000],
# 
#                 [1831, 266, 1000],
#                 [1831, 269, 1000],
# 
#                 [101457, 266, 1000],
#                 [101457, 269, 1000],
# 
#                 [101657, 269, 1000],
#                 [101657, 266, 1000],
# 
#                 [1841, 269, 1000],
#                 [1841, 266, 1000],
# 
#                 [101411, 269, 1000],
#                 [101411, 266, 1000],
#             ],
#             second_layer_model=model_id, #105233,
#             nsplits=5
#         ), level=-2
#     )
# 
#     res = cv.cross_val(model_id, -1)
#     print(model_id, res)


# model_id2 = qm.add_by_params(
#     QXgb(
# **
#
# {"alpha": 0.008, "booster": "gbtree", "colsample_bylevel": 0.6, "colsample_bytree": 1.0, "eta": 0.004,
#  "eval_metric": "logloss", "gamma": 0.2, "max_depth": 4, "num_boost_round": 2015, "objective": "binary:logistic", "subsample": 0.7}
#
#     ),
#     'hyperopt xgb',
# )


#
for data_id in [203]:
    for model_id in [1747, 101340,101340,101340,101331,101261,101655,101457,101411]:
        res = cv.cross_val(model_id, data_id, force=True, seed=1000)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1001)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1002)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1003)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1004)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1005)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1006)
        # res = cv.cross_val(model_id, data_id, force=True, seed=1007)
        print('--', res)





######1551 1698
# models = []
# for model_id, data_id in [[1255, 21], [1255, 23], [1255, 24], [1255, 26], [1255, 27], [1395, 42], [1395, 43], [1255, 43]]:
#
#     model_id_added = qm.add_by_params(
#         QAvg(
#             [
#                 [model_id, data_id, 1000],
#                 [model_id, data_id, 1001],
#                 [model_id, data_id, 1002],
#                 [model_id, data_id, 1003],
#                 [model_id, data_id, 1004],
#                 [model_id, data_id, 1005],
#                 [model_id, data_id, 1006],
#                 [model_id, data_id, 1007],
#             ]
#         ), level=-2
#     )
#     print(model_id_added)
#     models += [[model_id_added, -1]]
#
# model_id = qm.add_by_params(
#     QAvg(models), level=-3
# )
#
# res = cv.cross_val(model_id, -1)
# print(model_id, res)

#
#
#
# models = []
# for model_id, data_id in [[1747, 66], [1395, 66]]:
#
#     model_id_added = qm.add_by_params(
#         QAvg(
#             [
#                 [model_id, data_id, 1000],
#                 [model_id, data_id, 1001],
#                 [model_id, data_id, 1002],
#                 [model_id, data_id, 1003],
#                 [model_id, data_id, 1004],
#                 [model_id, data_id, 1005],
#                 [model_id, data_id, 1006],
#                 [model_id, data_id, 1007],
#             ]
#         ), level=-2
#     )
#     print(model_id_added)
#     models += [[model_id_added, -1]]
#
# model_id = qm.add_by_params(
#     QAvg(models), level=-3
# )

# res = cv.cross_val(model_id, -1)
#res = qm.qpredict(model_id, -1)
#print(model_id, res)


#
# qm.qpredict(1738,-1)
#
#



#
# model_id2 = qm.add_by_params(
#     QNN(**{"epochs": 200}),
#     'hyperopt nn1',
# )
#
# print(model_id2)
# #
# for data_id in [269]:
#     for model_id in [model_id2]:
#         res = cv.cross_val(model_id, data_id, force=True, seed=1000)