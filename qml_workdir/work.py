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

#qm.qpredict(109639,-1)
# qm.qpredict(108994,-1)
# qm.qpredict(109045,-1)
# qm.qpredict(108985,-1)

# 108994
# 109045
# 108985

# model_id = qm.add_by_params(
#     QAvg(
#         [
#             # [101340, 69, 1000],
#             # [101340, 69, 1001],
#             # [101340, 69, 1002],
#             # [101340, 69, 1003],
#             # [101340, 69, 1004],
#             # [101340, 69, 1005],
#             # [101340, 69, 1006],
#             # [101340, 69, 1007],
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
# cv.cross_val(model_id,-1)
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

# for data_id in [70]:
#     for model_id in [1747]:
#         res = cv.cross_val(model_id, data_id, force=True,seed=1000)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1001)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1002)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1003)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1004)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1005)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1006)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1007)
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
#             [1747, 69, 1008],
#     [101340, 69, 1000],
# [101283, 69, 1000],
# [101283, 69, 1001],
# [101283, 69, 1002],
# [101283, 69, 1003],
# [101283, 69, 1004],
# [101283, 69, 1005],
# [101283, 69, 1006],
# [101283, 69, 1007],
#
#              # [1747, 69, 1009],
#              # [1747, 69, 1010],
#              # [1747, 69, 1011],
#              # [1747, 69, 1012],
#              # [1747, 69, 1013],
#              # [1747, 69, 1014],
#              # [1747, 69, 1015],
#         ]
#
#     ), level=-2
# )
#
# res = cv.cross_val(model_id, -1)
# print(model_id, res)
#
# #


if __name__ == '__main__':



    model_id = qm.add_by_params(
        QAvg(
            [
                [101340, 69, 1000],
                [101340, 69, 1001],
                [101340, 69, 1002],
                [101340, 69, 1003],
                [101340, 69, 1004],
                [101340, 69, 1005],
                [101340, 69, 1006],
                [101340, 69, 1007],
            ]
        ), level=-2
    )

    res = cv.cross_val(model_id, -1)
    print(model_id, res)
    qm.qpredict(model_id, -1)


    model_id = qm.add_by_params(
        QAvg(
            [
                [101340, 300, 1000],
                [101340, 300, 1001],
                [101340, 300, 1002],
                [101340, 300, 1003],
                [101340, 300, 1004],
                [101340, 300, 1005],
                [101340, 300, 1006],
                [101340, 300, 1007],
            ]
        ), level=-2
    )

    res = cv.cross_val(model_id, -1)
    print(model_id, res)
    qm.qpredict(model_id, -1)




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

# print(model_id2)
# #
# for data_id in [69]:
#     for model_id in [101340]:
#         #res = cv.cross_val(model_id, data_id, force=True, seed=1000)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1001)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1002)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1003)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1004)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1005)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1006)
#         res = cv.cross_val(model_id, data_id, force=True, seed=1007)
#         print('--', res)





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