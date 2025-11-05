"""
Model Training and Prediction Module for Detox
Implements Logistic Regression for toxicity classification
"""

from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, round as spark_round
import config


class ToxicityClassifier:
    """
    Class to handle toxicity classification model
    """
    
    def __init__(self):
        self.model = None
        self.metrics = {}
    
    def train_model(self, train_df: DataFrame) -> LogisticRegressionModel:
        """
        Train Logistic Regression model
        
        Args:
            train_df: Training DataFrame with features and label
            
        Returns:
            Trained model
        """
        print("[INFO] Training Logistic Regression model...")
        
        # Initialize Logistic Regression
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=config.MAX_ITERATIONS,
            regParam=config.REGULARIZATION_PARAM,
            elasticNetParam=0.0,  # L2 regularization
            family="binomial"
        )
        
        # Train model
        self.model = lr.fit(train_df)
        
        print("[INFO] Model training complete")
        print(f"[INFO] Number of iterations: {self.model.summary.totalIterations}")
        
        return self.model
    
    def evaluate_model(self, test_df: DataFrame) -> dict:
        """
        Evaluate model performance
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print("[INFO] Evaluating model...")
        
        # Make predictions
        predictions = self.model.transform(test_df)
        
        # Binary Classification Evaluator
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        
        # Multiclass Evaluators
        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        # Calculate metrics
        self.metrics = {
            "auc": evaluator_auc.evaluate(predictions),
            "auc_pr": evaluator_pr.evaluate(predictions),
            "accuracy": evaluator_accuracy.evaluate(predictions),
            "precision": evaluator_precision.evaluate(predictions),
            "recall": evaluator_recall.evaluate(predictions),
            "f1_score": evaluator_f1.evaluate(predictions)
        }
        
        # Print metrics
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"AUC-ROC:          {self.metrics['auc']:.4f}")
        print(f"AUC-PR:           {self.metrics['auc_pr']:.4f}")
        print(f"Accuracy:         {self.metrics['accuracy']:.4f}")
        print(f"Precision:        {self.metrics['precision']:.4f}")
        print(f"Recall:           {self.metrics['recall']:.4f}")
        print(f"F1 Score:         {self.metrics['f1_score']:.4f}")
        print(f"{'='*60}\n")
        
        return self.metrics
    
    def predict(self, df: DataFrame) -> DataFrame:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print("[INFO] Making predictions...")
        
        predictions = self.model.transform(df)
        
        # Extract probability of toxic class (index 1) - use UDF for sparse vectors
        from pyspark.ml.linalg import VectorUDT
        from pyspark.sql.types import DoubleType
        from pyspark.sql.functions import udf
        
        def extract_prob(probability):
            """Extract probability for class 1 (toxic)"""
            if probability is not None:
                try:
                    return float(probability[1])
                except:
                    return 0.0
            return 0.0
        
        extract_prob_udf = udf(extract_prob, DoubleType())
        
        predictions = predictions.withColumn(
            "toxicity_score",
            spark_round(extract_prob_udf(col("probability")), 4)
        )
        
        # Add toxicity level based on score
        predictions = predictions.withColumn(
            "toxicity_level",
            when(col("toxicity_score") >= config.SEVERE_TOXICITY_THRESHOLD, "SEVERE")
            .when(col("toxicity_score") >= config.HIGH_TOXICITY_THRESHOLD, "HIGH")
            .when(col("toxicity_score") >= config.TOXICITY_THRESHOLD, "MODERATE")
            .otherwise("LOW")
        )
        
        print("[INFO] Predictions complete")
        
        return predictions
    
    def save_model(self, path: str):
        """
        Save trained model
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print(f"[INFO] Saving model to: {path}")
        self.model.write().overwrite().save(path)
        print("[INFO] Model saved successfully")
    
    def load_model(self, path: str):
        """
        Load trained model
        
        Args:
            path: Path to load the model from
        """
        print(f"[INFO] Loading model from: {path}")
        self.model = LogisticRegressionModel.load(path)
        print("[INFO] Model loaded successfully")
