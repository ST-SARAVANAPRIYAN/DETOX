"""
Main Application for Detox - Chat Message Toxicity Detector
ADI1302-SPARK SCALA FUNDAMENTALS
NAME : SARAVANA PRIYAN S T
REG NO : 927623BAD100

This is the main entry point for the Detox toxicity detection system.
Run this script to execute the complete pipeline with Spark Web UI monitoring.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import config
from data_ingestion import DataIngestion
from preprocessing import TextPreprocessor
from model import ToxicityClassifier
from user_analysis import UserToxicityAnalyzer
import time


def create_spark_session():
    """
    Create and configure Spark session with Web UI enabled
    """
    print(f"\n{'='*60}")
    print(f"DETOX - CHAT MESSAGE TOXICITY DETECTOR")
    print(f"{'='*60}")
    print(f"Student: SARAVANA PRIYAN S T")
    print(f"Reg No: 927623BAD100")
    print(f"Course: ADI1302-SPARK SCALA FUNDAMENTALS")
    print(f"{'='*60}\n")
    
    print("[INFO] Creating Spark Session...")
    print(f"[INFO] Spark Web UI will be available at: http://localhost:{config.SPARK_UI_PORT}")
    
    spark = SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master(config.SPARK_MASTER) \
        .config("spark.executor.memory", config.SPARK_EXECUTOR_MEMORY) \
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY) \
        .config("spark.ui.port", config.SPARK_UI_PORT) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"[INFO] Spark Session created successfully")
    print(f"[INFO] Spark Version: {spark.version}")
    print(f"[INFO] Master: {spark.sparkContext.master}")
    print(f"[INFO] Application ID: {spark.sparkContext.applicationId}\n")
    
    return spark


def main():
    """
    Main execution function
    """
    start_time = time.time()
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Step 1: Data Ingestion
        print(f"\n{'='*60}")
        print("STEP 1: DATA INGESTION")
        print(f"{'='*60}\n")
        
        data_ingestion = DataIngestion(spark)
        df = data_ingestion.read_csv_data(config.INPUT_DATA_PATH)
        df = data_ingestion.validate_data(df)
        stats = data_ingestion.get_data_statistics(df)
        
        # Cache the dataframe for better performance
        df.cache()
        
        # Step 2: Data Preprocessing
        print(f"\n{'='*60}")
        print("STEP 2: DATA PREPROCESSING")
        print(f"{'='*60}\n")
        
        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_text(df)
        df = preprocessor.create_label_column(df, label_col="toxic")
        df = preprocessor.fit_transform_features(df)
        
        # Step 3: Split data
        print(f"\n{'='*60}")
        print("STEP 3: TRAIN-TEST SPLIT")
        print(f"{'='*60}\n")
        
        train_df, test_df = df.randomSplit(
            [config.TRAIN_TEST_SPLIT_RATIO, 1 - config.TRAIN_TEST_SPLIT_RATIO],
            seed=config.RANDOM_SEED
        )
        
        print(f"[INFO] Training set: {train_df.count()} records")
        print(f"[INFO] Test set: {test_df.count()} records")
        
        train_df.cache()
        test_df.cache()
        
        # Step 4: Model Training
        print(f"\n{'='*60}")
        print("STEP 4: MODEL TRAINING")
        print(f"{'='*60}\n")
        
        classifier = ToxicityClassifier()
        # Train the model on features
        model = classifier.train_model(train_df)
        
        # Create complete pipeline (feature extraction + model) for production
        classifier.set_complete_pipeline(preprocessor.pipeline_model, model)
        
        # Step 5: Model Evaluation
        print(f"\n{'='*60}")
        print("STEP 5: MODEL EVALUATION")
        print(f"{'='*60}\n")
        
        metrics = classifier.evaluate_model(test_df)
        
        # Step 6: Make Predictions on Full Dataset
        print(f"\n{'='*60}")
        print("STEP 6: PREDICTIONS")
        print(f"{'='*60}\n")
        
        predictions_df = classifier.predict(df)
        predictions_df.cache()
        
        # Step 7: User-Level Analysis
        print(f"\n{'='*60}")
        print("STEP 7: USER-LEVEL TOXICITY ANALYSIS")
        print(f"{'='*60}\n")
        
        analyzer = UserToxicityAnalyzer()
        user_aggregates = analyzer.aggregate_user_toxicity(predictions_df)
        user_stats = analyzer.get_user_statistics(user_aggregates)
        
        # Get top toxic users
        top_toxic_users = analyzer.get_top_toxic_users(user_aggregates, top_n=20)
        
        print("[INFO] Top 20 Most Toxic Users:")
        top_toxic_users.select(
            "user_id", 
            "total_messages", 
            "avg_toxicity_score", 
            "user_toxicity_level",
            "toxic_messages_count"
        ).show(20, truncate=False)
        
        # Create detailed message report
        message_report = analyzer.create_detailed_message_report(predictions_df)
        
        # Step 8: Save Results
        print(f"\n{'='*60}")
        print("STEP 8: SAVING RESULTS")
        print(f"{'='*60}\n")
        
        # Save predictions
        print(f"[INFO] Saving predictions to: {config.OUTPUT_PREDICTIONS_PATH}")
        predictions_df.select(
            "id",
            F.substring("id", 1, 8).alias("user_id"),
            "comment_text",
            "toxicity_score",
            "toxicity_level",
            "prediction",
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ).coalesce(1).write.mode("overwrite").csv(
            config.OUTPUT_PREDICTIONS_PATH,
            header=True
        )
        
        # Save user aggregates
        print(f"[INFO] Saving user toxicity levels to: {config.OUTPUT_USER_AGGREGATES_PATH}")
        user_aggregates.coalesce(1).write.mode("overwrite").csv(
            config.OUTPUT_USER_AGGREGATES_PATH,
            header=True
        )
        
        # Save model
        classifier.save_model(config.MODEL_PATH)
        
        # Step 9: Final Summary
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        execution_time = time.time() - start_time
        
        print(f"\n✓ Data Processing: {stats['total_records']} records processed")
        print(f"✓ Model Training: Completed with {metrics['accuracy']:.4f} accuracy")
        print(f"✓ Predictions: Generated for all messages")
        print(f"✓ User Analysis: {user_stats['total_users']} users analyzed")
        print(f"✓ Outputs Saved: Predictions and user aggregates exported")
        print(f"✓ Model Saved: {config.MODEL_PATH}")
        print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
        print(f"\n{'='*60}")
        print(f"✓ DETOX PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}\n")
        
        print(f"[INFO] Spark Web UI: http://localhost:{config.SPARK_UI_PORT}")
        print(f"[INFO] Press Ctrl+C to stop the application and close Spark session\n")
        
        # Keep the application running to access Spark Web UI
        print("[INFO] Application is running. You can now access Spark Web UI.")
        print("[INFO] Press Ctrl+C to exit...\n")
        
        # Keep alive
        input("Press Enter to exit and stop Spark session...")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n[INFO] Stopping Spark session...")
        spark.stop()
        print("[INFO] Spark session stopped. Goodbye!\n")


def run_detox_pipeline():
    """Wrapper function to run the pipeline with timing"""
    print("\n" + "="*80)
    print("DETOX - Chat Message Toxicity Detection System")
    print("="*80 + "\n")
    
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"{'='*80}\n")
    return True

if __name__ == "__main__":
    run_detox_pipeline()
