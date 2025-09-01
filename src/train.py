# src/train.py
import os
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def train_and_select_model():
    """
    Optimized training script:
    - Uses TrainValidationSplit (faster than CrossValidator)
    - Reduces hyperparameter grid size
    - Enables caching and parallelism
    - Logs best model with MLflow
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    spark = (
        SparkSession.builder
        .appName("ModelTraining")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")  # reduced for small dataset
        .getOrCreate()
    )

    # Load preprocessor and training data
    preprocessor = PipelineModel.load("models/preprocessor")
    train_df = spark.read.parquet("data/prepared/train.parquet")

    # Cache transformed training data
    processed_train_df = preprocessor.transform(train_df).cache()

    # Define models and reduced hyperparameter grids
    lr = LogisticRegression(featuresCol="features", labelCol="Survived")
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])      # coarse search
        .addGrid(lr.elasticNetParam, [0.0, 0.5])  # L2 + ElasticNet
        .build()
    )

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")
    dt_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [3, 7])    # shallower trees
        .addGrid(dt.impurity, ["gini", "entropy"])
        .build()
    )

    rf = RandomForestClassifier(featuresCol="features", labelCol="Survived")
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50])   # smaller forest
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )

    gbt = GBTClassifier(featuresCol="features", labelCol="Survived")
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [20, 50])  # fewer boosting rounds
        .addGrid(gbt.maxDepth, [3, 5])
        .build()
    )

    models = {
        "LogisticRegression": (lr, lr_grid),
        "DecisionTree": (dt, dt_grid),
        "RandomForest": (rf, rf_grid),
        "GBT": (gbt, gbt_grid),
    }

    best_model = None
    best_score = -1.0

    evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")

    with mlflow.start_run(run_name="Optimized SparkML Model Selection") as parent_run:
        for model_name, (model, param_grid) in models.items():
            with mlflow.start_run(run_name=f"Tune_{model_name}", nested=True):
                tvs = TrainValidationSplit(
                    estimator=model,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    trainRatio=0.8,
                    parallelism=4,  # run param sets in parallel
                )

                print(f"Training {model_name} with reduced param grid...")
                tvs_model = tvs.fit(processed_train_df)

                score = evaluator.evaluate(tvs_model.bestModel.transform(processed_train_df))
                mlflow.log_metric("val_auc", score)

                best_params = {
                    p.name: v for p, v in tvs_model.bestModel.extractParamMap().items()
                    if p in param_grid[0]
                }
                mlflow.log_params(best_params)

                if score > best_score:
                    best_score = score
                    best_model = tvs_model.bestModel
                    mlflow.set_tag("best_model_type", model_name)

        mlflow.log_metric("best_overall_auc", best_score)

        # Save the best model locally
        output_path = "models/classifier"
        os.makedirs("models", exist_ok=True)
        best_model.write().overwrite().save(output_path)
        print(f"\nBest model ({best_model.__class__.__name__}) saved to {output_path}")

        # Register the best model in MLflow registry
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="champion-classifier",
            registered_model_name="TitanicChampionClassifier",
        )
        print("Best model registered in MLflow Model Registry.")

    spark.stop()


if __name__ == "__main__":
    train_and_select_model()
