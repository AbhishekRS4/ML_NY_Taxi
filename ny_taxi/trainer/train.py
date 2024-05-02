import os
import mlflow
import logging
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold

from ny_taxi.utils.metrics import compute_metrics
from ny_taxi.modeling.pipeline import get_pipeline
from ny_taxi.dataset.data_loader import data_loader
from ny_taxi.dataset.data_transformer import transform
from ny_taxi.config.config import PipelineConfig, FeatureTargetConfig, DataLoaderConfig


def train_pipeline(
    config_train_loader: DataLoaderConfig,
    config_test_loader: DataLoaderConfig,
    config_pipeline: PipelineConfig,
) -> None:
    config_feature_target = FeatureTargetConfig()
    categorical = config_feature_target.categorical
    numerical = config_feature_target.numerical
    target = config_feature_target.target

    # load and transform train data
    logging.info(f"config train dataloader: {config_train_loader}")
    df_train = data_loader(config_train_loader)
    df_train = transform(df_train)
    logging.info("completed loading and applying transformation on train data")
    X_train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    Y_train = df_train[target].values
    logging.info(f"num trips in train set: {df_train.shape[0]}")

    # load and transform test data
    logging.info(f"config test dataloader: {config_test_loader}")
    df_test = data_loader(config_test_loader)
    df_test = transform(df_test)
    logging.info("completed loading and applying transformation on test data")
    X_test_dicts = df_test[categorical + numerical].to_dict(orient="records")
    Y_test = df_test[target].values
    logging.info(f"num trips in test set: {df_test.shape[0]}")

    # setup model pipeline
    logging.info(f"config model pipeline: {config_pipeline}")
    pipeline, pipeline_params = get_pipeline(config_pipeline)

    # setup grid search with k-fold cross validation
    k_fold_cv = KFold(n_splits=5, shuffle=True, random_state=7)
    grid_cv = GridSearchCV(
        pipeline,
        pipeline_params,
        scoring="neg_root_mean_squared_error",
        cv=k_fold_cv,
    )

    # train the grid search model
    grid_cv.fit(X_train_dicts, Y_train)

    # get the cross validation score and the params for the best estimator
    cv_best_estimator = grid_cv.best_estimator_
    cv_best_rmse = grid_cv.best_score_
    cv_best_params = grid_cv.best_params_

    logging.info(f"best estimator: {cv_best_estimator}")
    logging.info(f"best estimator score: {cv_best_rmse}")

    # compute predictions for train and test sets with the best estimator
    Y_train_pred = cv_best_estimator.predict(X_train_dicts)
    Y_test_pred = cv_best_estimator.predict(X_test_dicts)

    # compute metrics for the train and test sets with the best estimator
    train_rmse, train_r2 = compute_metrics(Y_train, Y_train_pred)
    test_rmse, test_r2 = compute_metrics(Y_test, Y_test_pred)

    logging.info(f"train_rmse: {train_rmse:.4f}, train_r2: {train_r2:.4f}")
    logging.info(f"test_rmse: {test_rmse:.4f}, test_r2: {test_r2:.4f}")

    # use mlflow and log models, params and metrics
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("ny_taxi")

    experiment = mlflow.get_experiment_by_name("ny_taxi")

    logging.info(f"started mlflow logging for the best estimator")
    model_log_str = config_pipeline.regressor_type
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.set_tag("developer", "abhishek_r_s")
        mlflow.sklearn.log_model(cv_best_estimator, model_log_str)
        mlflow.log_params(cv_best_params)
        mlflow.log_param("train_data", config_train_loader.all_files)
        mlflow.log_param("test_data", config_test_loader.all_files)
        mlflow.log_metric("cv_best_rmse", cv_best_rmse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
    # end mlflow logging
    mlflow.end_run()
    logging.info("completed mlflow logging for the best estimator")
    return
