"""
Model Training and Prediction Module for Detox
Implements Logistic Regression for toxicity classification
"""

from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, when, round as spark_round
import config


class ToxicityClassifier:
    """
    Class to handle toxicity classification model
    """
    
    def __init__(self):
        self.model = None
        self.complete_pipeline = None  # Complete pipeline including feature extraction
        self.metrics = {}
    
    def train_model(self, train_df: DataFrame) -> LogisticRegressionModel:
        """
        Train Logistic Regression model with class weight balancing
        
        Args:
            train_df: Training DataFrame with features and label
            
        Returns:
            Trained model
        """
        print("[INFO] Training Logistic Regression model...")
        
        # Calculate class distribution for balancing
        label_counts = train_df.groupBy("label").count().collect()
        total = sum([row['count'] for row in label_counts])
        
        # Calculate class weights (inverse frequency)
        class_weights = {}
        for row in label_counts:
            label = row['label']
            count = row['count']
            # Weight = total / (n_classes * count)
            class_weights[label] = total / (2.0 * count)
        
        print(f"[INFO] Class distribution:")
        for row in label_counts:
            label = "Toxic" if row['label'] == 1.0 else "Non-toxic"
            print(f"  {label}: {row['count']} ({row['count']/total*100:.1f}%)")
        
        print(f"[INFO] Class weights: {class_weights}")
        
        # Initialize Logistic Regression with class weights
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=config.MAX_ITERATIONS,
            regParam=config.REGULARIZATION_PARAM,
            elasticNetParam=0.0,  # L2 regularization
            family="binomial",
            weightCol="classWeight"  # Use class weights
        )
        
        # Add class weight column
        from pyspark.sql.functions import when
        train_df_weighted = train_df.withColumn(
            "classWeight",
            when(col("label") == 1.0, class_weights.get(1.0, 1.0))
            .otherwise(class_weights.get(0.0, 1.0))
        )
        
        # Add class weight column
        from pyspark.sql.functions import when
        train_df_weighted = train_df.withColumn(
            "classWeight",
            when(col("label") == 1.0, class_weights.get(1.0, 1.0))
            .otherwise(class_weights.get(0.0, 1.0))
        )
        
        # Train model with weighted samples
        self.model = lr.fit(train_df_weighted)
        
        print("[INFO] Model training complete")
        print(f"[INFO] Number of iterations: {self.model.summary.totalIterations}")
        
        return self.model
    
    def set_complete_pipeline(self, feature_pipeline_model, lr_model):
        """
        Create complete pipeline by combining feature extraction + trained model
        
        Args:
            feature_pipeline_model: Fitted feature extraction pipeline
            lr_model: Trained LogisticRegression model
        """
        print("[INFO] Creating complete pipeline (feature extraction + model)")
        
        # Get all stages from feature pipeline + LR model
        all_stages = feature_pipeline_model.stages + [lr_model]
        
        # Create a new pipeline with all stages
        self.complete_pipeline = PipelineModel(all_stages)
        
        print("[SUCCESS] ✓ Complete pipeline created")
    
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
        Save the complete pipeline model
        
        Args:
            path: Path to save the model
        """
        print(f"[INFO] Saving model to {path}...")
        
        if self.complete_pipeline is not None:
            print("[INFO] Saving complete PipelineModel (includes feature extraction + classifier)")
            self.complete_pipeline.write().overwrite().save(path)
            print(f"[SUCCESS] ✓ Complete PipelineModel saved to {path}")
        elif self.model is not None:
            print("[WARNING] Saving only LogisticRegressionModel (feature extraction not included)")
            self.model.write().overwrite().save(path)
            print(f"[SUCCESS] ✓ LogisticRegressionModel saved to {path}")
        else:
            raise ValueError("No model to save. Train the model first.")
    
    def load_model(self, path: str):
        """
        Load trained model
        
        Args:
            path: Path to load the model from
        """
        print(f"[INFO] Loading model from: {path}")
        self.model = LogisticRegressionModel.load(path)
        print("[INFO] Model loaded successfully")
