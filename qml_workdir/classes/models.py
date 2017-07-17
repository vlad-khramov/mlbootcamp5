import qml_workdir.classes.config
from qml.models import QModels, QXgb, QXgb2
from sklearn.linear_model import LogisticRegression

qm = QModels()


qm.add(1,
    QXgb(
        booster='gblinear',
        objective='binary:logistic',
        eval_metric='logloss',
        eta=0.1,#learn_rate
        max_depth=3,

        num_boost_round=1000

    ),
    'simple xgb linear'
)


qm.add(2,
    QXgb(
        booster='gblinear',
        objective='binary:logistic',
        eval_metric='logloss',
        subsample=0.5,
        eta=0.1,#learn_rate
        max_depth=3,

        num_boost_round=100

    ),
    'simple xgb linear'
)

qm.add(3,
   LogisticRegression(
       n_jobs=-1
   ),
    'simple xgb'
)

qm.add(5,
    QXgb(
        booster='gbtree',
        objective='multi:softprob',
        eval_metric='mlogloss',
        subsample=0.5,
        eta=0.001,#learn_rate
        max_depth=3,
        num_class=3,

        num_boost_round=3400

    ),
    'simple xgb'
)



qm.add(1000,
    QXgb(),
    'dummy'
)

