#!/usr/bin/env python3
"""
Quick test to verify model can be loaded
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Model path
model_path = "/home/saravana/projects/ssfproject/models/toxicity_model"

print(f"Testing model load from: {model_path}")
print(f"Path exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print("\nCreating Spark session...")
    spark = SparkSession.builder \
        .appName("ModelTest") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session created")
    
    print("\nLoading model...")
    try:
        model = PipelineModel.load(model_path)
        print("✓ Model loaded successfully!")
        print(f"Model stages: {len(model.stages)}")
        
        # Test prediction
        print("\nTesting prediction...")
        test_df = spark.createDataFrame([("test message",)], ["text"])
        result = model.transform(test_df)
        print("✓ Prediction successful!")
        result.select("text", "prediction", "probability").show(truncate=False)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
else:
    print("✗ Model path does not exist!")
