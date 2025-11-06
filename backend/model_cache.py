"""
Model Caching System - Quick Win #1
====================================
Load model once at startup instead of per-request
Reduces latency from seconds to milliseconds
"""

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
import os
import sys
import logging
from threading import Lock

# Add parent directory to path to import toxicity_lexicon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toxicity_lexicon import get_toxic_word_count

logger = logging.getLogger(__name__)

class ModelCache:
    """Singleton pattern for model caching"""
    _instance = None
    _lock = Lock()
    _model = None
    _spark = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, model_path: str, app_name: str = "ToxicityDetectionAPI"):
        """Initialize Spark session and load model once"""
        if self._model is not None:
            print("[INFO] Model already loaded, skipping initialization")
            logger.info("Model already loaded, skipping initialization")
            return
        
        try:
            print(f"[INFO] Attempting to load model from: {model_path}")
            print(f"[INFO] Path exists: {os.path.exists(model_path)}")
            
            # Create Spark session (if not exists)
            if self._spark is None:
                self._spark = SparkSession.builder \
                    .appName(app_name) \
                    .config("spark.driver.memory", "2g") \
                    .config("spark.executor.memory", "2g") \
                    .getOrCreate()
                
                # Reduce logging verbosity
                self._spark.sparkContext.setLogLevel("ERROR")
                print("[SUCCESS] ✓ Spark session created")
                logger.info("✓ Spark session created")
            
            # Load model
            if os.path.exists(model_path):
                print(f"[INFO] Loading model from {model_path}...")
                # Try loading as LogisticRegressionModel first (single model)
                try:
                    self._model = LogisticRegressionModel.load(model_path)
                    print(f"[SUCCESS] ✓ Model loaded as LogisticRegressionModel!")
                    logger.info(f"✓ Model loaded from {model_path}")
                except:
                    # If that fails, try as PipelineModel
                    print(f"[INFO] Trying to load as PipelineModel...")
                    self._model = PipelineModel.load(model_path)
                    print(f"[SUCCESS] ✓ Model loaded as PipelineModel!")
                    logger.info(f"✓ Pipeline model loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
                
        except Exception as e:
            print(f"[ERROR] ✗ Failed to initialize model: {e}")
            logger.error(f"✗ Failed to initialize model: {e}")
            raise
    
    def get_model(self):
        """Get cached model"""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._model
    
    def get_spark(self):
        """Get Spark session"""
        if self._spark is None:
            raise RuntimeError("Spark not initialized. Call initialize() first.")
        return self._spark
    
    def predict_single(self, text: str) -> dict:
        """
        Predict toxicity for a single message (with lexicon features)
        Returns: {prediction: 0/1, toxicity_score: float, level: str, latency_ms: float}
        """
        import time
        import re
        start_time = time.time()
        
        try:
            # Clean text (same as preprocessing.py)
            cleaned = text.lower()
            cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
            cleaned = re.sub(r'\S+@\S+', '', cleaned)
            cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
            cleaned = ' '.join(cleaned.split())
            
            # Get lexicon features
            lexicon_stats = get_toxic_word_count(cleaned)
            
            # Create DataFrame with cleaned_text and lexicon features
            df = self._spark.createDataFrame([
                (
                    cleaned,
                    lexicon_stats['extreme_toxic_count'],
                    lexicon_stats['high_toxic_count'],
                    lexicon_stats['medium_toxic_count'],
                    lexicon_stats['low_toxic_count'],
                    lexicon_stats['total_toxic_count'],
                    float(lexicon_stats['toxic_word_ratio']),
                    float(lexicon_stats['severity_score'])
                )
            ], [
                "cleaned_text",
                "extreme_toxic_count",
                "high_toxic_count",
                "medium_toxic_count",
                "low_toxic_count",
                "total_toxic_count",
                "toxic_word_ratio",
                "severity_score"
            ])
            
            # Predict
            result = self._model.transform(df)
            
            # Extract results
            row = result.select("prediction", "probability").first()
            prediction = int(row.prediction)
            prob = float(row.probability[1])  # Probability of toxic class
            
            # Determine toxicity level
            if prob >= 0.9:
                level = "VERY_HIGH"
            elif prob >= 0.7:
                level = "HIGH"
            elif prob >= 0.5:
                level = "MODERATE"
            elif prob >= 0.3:
                level = "LOW"
            else:
                level = "MINIMAL"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "prediction": "Toxic" if prediction == 1 else "Non-toxic",
                "toxicity_score": round(prob, 4),
                "level": level,
                "latency_ms": round(latency_ms, 2),
                "lexicon_severity": round(lexicon_stats['severity_score'], 2),
                "success": True
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed: {e}")
            return {
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
                "success": False
            }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict toxicity for multiple messages at once
        Much faster than individual predictions
        
        Args:
            texts: List of text strings
        
        Returns:
            List of prediction dictionaries
        """
        import time
        import re
        start_time = time.time()
        
        try:
            # Clean all texts (same as preprocessing.py)
            cleaned_texts = []
            for text in texts:
                cleaned = text.lower()
                cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
                cleaned = re.sub(r'\S+@\S+', '', cleaned)
                cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
                cleaned = ' '.join(cleaned.split())
                cleaned_texts.append(cleaned)
            
            # Create DataFrame with cleaned_text column (pipeline expects this)
            df = self._spark.createDataFrame([(text,) for text in cleaned_texts], ["cleaned_text"])
            
            # Predict all at once
            results = self._model.transform(df)
            
            # Extract results
            predictions = []
            original_texts = texts  # Keep original for display
            for i, row in enumerate(results.select("cleaned_text", "prediction", "probability").collect()):
                prediction = int(row.prediction)
                prob = float(row.probability[1])
                
                # Determine level
                if prob >= 0.9:
                    level = "VERY_HIGH"
                elif prob >= 0.7:
                    level = "HIGH"
                elif prob >= 0.5:
                    level = "MODERATE"
                elif prob >= 0.3:
                    level = "LOW"
                else:
                    level = "MINIMAL"
                
                original_text = original_texts[i]
                predictions.append({
                    "text": original_text[:50] + "..." if len(original_text) > 50 else original_text,
                    "prediction": "Toxic" if prediction == 1 else "Non-toxic",
                    "toxicity_score": round(prob, 4),
                    "level": level
                })
            
            latency_ms = (time.time() - start_time) * 1000
            avg_latency = latency_ms / len(texts)
            
            logger.info(f"✓ Batch prediction: {len(texts)} messages in {latency_ms:.2f}ms (avg: {avg_latency:.2f}ms)")
            
            return {
                "predictions": predictions,
                "total": len(predictions),
                "total_latency_ms": round(latency_ms, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "success": True
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Batch prediction failed: {e}")
            return {
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
                "success": False
            }
    
    def get_stats(self) -> dict:
        """Get model and system statistics"""
        return {
            "model_loaded": self._model is not None,
            "spark_active": self._spark is not None and not self._spark._jsc.sc().isStopped(),
            "spark_version": self._spark.version if self._spark else None,
            "model_stages": len(self._model.stages) if self._model else 0
        }


# Global instance
model_cache = ModelCache()
