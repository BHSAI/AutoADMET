from pathlib import Path

import pandas as pd
import numpy as np
from flaml import AutoML
import pickle
import vnn_estimator_flaml
from sklearn.preprocessing import MinMaxScaler
import logging
import itertools as it
from sklearn.metrics import cohen_kappa_score


USE_PREFIT = False

datasets = [
    "ames",
    "dili",
    "hepatotoxicity",
    "hlm",
    "mmp",
]

featurizations = [
    "morgan_fp",
    "mordred_desc",
]

time_limit = 240

seed = 7654321


def cohen_kappa(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    config=None,
    groups_val=None,
    groups_train=None,
):
    y_pred = estimator.predict(X_val)
    return 1 - cohen_kappa_score(y_pred, y_val), {}


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")
    for dataset, featurization in it.product(datasets, featurizations):
        logger.info(f"Working on {dataset} {featurization} data")

        logger.info(f"{dataset} {featurization} - Loading in preprocessed data")
        train_csv = f"data/preprocessed/{dataset}/{featurization}.train.csv"
        test_csv = f"data/preprocessed/{dataset}/{featurization}.test.csv"
        train = pd.read_csv(train_csv)
        train.columns = train.columns.astype(np.str_)
        test = pd.read_csv(test_csv)
        test.columns = test.columns.astype(np.str_)

        class_col = "CLASS"
        feature_cols = [col for col in train.columns if "FEATURE_" in col]
        X_train = train[feature_cols]
        y_train = train[class_col].to_numpy()
        X_test = test[feature_cols]
        y_test = test[class_col].to_numpy()

        time_limit_str = (
            f"{time_limit}min" if time_limit is not None else "no_time_limit"
        )
        config_id_str = f"{dataset}.{featurization}.{time_limit_str}"
        
        Path("flaml/logs").mkdir(exist_ok=True)
        settings = {
            "time_budget": time_limit * 60,  # total running time in seconds
            "task": "classification",  # task type
            "log_file_name": f"flaml/logs/{config_id_str}.log",  # flaml log file
            "seed": seed,  # random seed
            "ensemble": True,
            "metric": cohen_kappa,
            "eval_method": "cv",
            "log_type": "all",
        }

        automl = AutoML()

        if featurization == "morgan_fp":
            settings["estimator_list"] = (
                [
                    "lgbm",
                    "rf",
                    "xgboost",
                    "extra_tree",
                    "xgb_limitdepth",
                    "sgd",
                    "catboost",
                    "lrl1",
                    "vnn",
                ],
            )
            automl.add_learner("vnn", vnn_estimator_flaml.VNNEstimator)
        else:
            scaler = MinMaxScaler()
            scaler.fit(pd.concat([X_train, X_test]))

            X_train[X_train.columns] = scaler.transform(X_train)
            X_test[X_test.columns] = scaler.transform(X_test)

        logger.info(f"{dataset} {featurization} - Fitting AutoGluon predictor")
        predictor_path = f'flaml/models/model.{config_id_str}.pkl'
        if not USE_PREFIT:
            automl.fit(
                X_train=X_train,
                y_train=y_train,
                **settings,
            )
            Path("flaml/models").mkdir(exist_ok=True)
            with open(predictor_path, 'wb') as f:
                pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(predictor_path, 'rb') as f:
                automl = pickle.load(f)


if __name__ == "__main__":
    main()
