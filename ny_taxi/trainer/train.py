import os
import mlflow
import numpy as np

from datetime import date
from prefect import flow, task, get_run_logger
from prefect.utilities.annotations import quote
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import GridSearchCV


from ny_taxi.utils.metrics import compute_metrics
from ny_taxi.dataset.data_loader import data_loader
from ny_taxi.dataset.data_transformer import transform
from ny_taxi.modeling.pipeline import get_pipeline, do_grid_search
from ny_taxi.config.config import (
    PipelineConfig,
    FeatureTargetConfig,
    DataLoaderConfig,
    TrainerConfig,
)


@task(retries=3, retry_delay_seconds=2)
def log_model_metrics(
    config_trainer: TrainerConfig,
    config_train_loader: DataLoaderConfig,
    config_test_loader: DataLoaderConfig,
    config_pipeline: PipelineConfig,
    grid_cv: GridSearchCV,
    train_rmse: float,
    train_r2: float,
    test_rmse: float,
    test_r2: float,
) -> None:
    logger = get_run_logger()
    # get the cross validation score and the params for the best estimator
    grid_cv_best_estimator = grid_cv.best_estimator_
    grid_cv_best_rmse = grid_cv.best_score_
    grid_cv_best_params = grid_cv.best_params_

    logger.info(f"best estimator score: {grid_cv_best_rmse}")

    # use mlflow and log models, params and metrics
    mlflow.set_tracking_uri(config_trainer.mlflow_tracking_uri)
    mlflow.set_experiment(config_trainer.experiment_name)

    experiment = mlflow.get_experiment_by_name(config_trainer.experiment_name)

    logger.info(f"started mlflow logging for the best estimator")
    model_log_str = config_pipeline.regressor_type
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.set_tag("developer", "abhishek_r_s")
        mlflow.sklearn.log_model(grid_cv_best_estimator, model_log_str)
        mlflow.log_param("regressor", config_pipeline.regressor_type)
        mlflow.log_params(grid_cv_best_params)
        mlflow.log_param("train_data", config_train_loader.all_files)
        mlflow.log_param("test_data", config_test_loader.all_files)
        mlflow.log_metric("grid_cv_best_rmse", grid_cv_best_rmse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
    # end mlflow logging
    mlflow.end_run()

    markdown__rmse_report = f"""# RMSE Report

        ## Summary

        NY Taxi Duration Prediction 

        ## RMSE {config_pipeline.regressor_type} Model

        | Region    | train RMSE | test RMSE | 
        |:----------|-----------:|----------:|
        | {date.today()} | {train_rmse:.4f} | | {test_rmse:.4f} | 
    """

    create_markdown_artifact(
        key="duration-model-report", markdown=markdown__rmse_report
    )

    logger.info("completed mlflow logging for the best estimator")
    return


@flow(validate_parameters=False)
def train_pipeline(
    config_train_loader: DataLoaderConfig,
    config_test_loader: DataLoaderConfig,
    config_pipeline: PipelineConfig,
    config_trainer: TrainerConfig,
) -> None:
    logger = get_run_logger()

    config_feature_target = FeatureTargetConfig()
    categorical = config_feature_target.categorical
    numerical = config_feature_target.numerical
    target = config_feature_target.target

    # load and transform train data
    df_train, config_train_loader = data_loader(config_train_loader)
    df_train = transform(df_train)
    logger.info(f"config train dataloader: {config_train_loader}")
    logger.info("completed loading and applying transformation on train data")
    X_train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    Y_train = df_train[target].values
    logger.info(f"num trips in train set: {df_train.shape[0]}")

    # load and transform test data
    df_test, config_test_loader = data_loader(config_test_loader)
    df_test = transform(df_test)
    logger.info(f"config test dataloader: {config_test_loader}")
    logger.info("completed loading and applying transformation on test data")
    X_test_dicts = df_test[categorical + numerical].to_dict(orient="records")
    Y_test = df_test[target].values
    logger.info(f"num trips in test set: {df_test.shape[0]}")

    # setup model pipeline
    logger.info(f"config model pipeline: {config_pipeline}")
    pipeline, pipeline_params = get_pipeline(config_pipeline)

    # do grid search
    # NOTE: use quote(param) for large params or data
    grid_cv = do_grid_search(
        pipeline, pipeline_params, quote(X_train_dicts), quote(Y_train)
    )

    grid_cv_best_estimator = grid_cv.best_estimator_
    logger.info(f"best estimator: {grid_cv_best_estimator}")

    # compute predictions for train and test sets with the best estimator
    Y_train_pred = grid_cv_best_estimator.predict(X_train_dicts)
    Y_test_pred = grid_cv_best_estimator.predict(X_test_dicts)

    # compute metrics for the train and test sets with the best estimator
    train_rmse, train_r2 = compute_metrics(Y_train, Y_train_pred)
    test_rmse, test_r2 = compute_metrics(Y_test, Y_test_pred)

    logger.info(f"train_rmse: {train_rmse:.4f}, train_r2: {train_r2:.4f}")
    logger.info(f"test_rmse: {test_rmse:.4f}, test_r2: {test_r2:.4f}")

    # log the best model from grid search cross validation and the metrics
    log_model_metrics(
        config_trainer,
        config_train_loader,
        config_test_loader,
        config_pipeline,
        grid_cv,
        train_rmse,
        train_r2,
        test_rmse,
        test_r2,
    )

    return
