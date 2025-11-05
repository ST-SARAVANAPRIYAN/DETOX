"""
Data Ingestion Module for Detox
Handles reading and initial validation of toxic comments dataset
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim, lower, regexp_replace
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import config


class DataIngestion:
    """
    Class to handle data ingestion from CSV files
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def read_csv_data(self, file_path: str):
        """
        Read CSV data with proper schema
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            PySpark DataFrame
        """
        print(f"[INFO] Reading data from: {file_path}")
        
        # Define schema for the dataset (with all 11 columns)
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("comment_text", StringType(), True),
            StructField("toxic", IntegerType(), True),
            StructField("severe_toxic", IntegerType(), True),
            StructField("obscene", IntegerType(), True),
            StructField("threat", IntegerType(), True),
            StructField("insult", IntegerType(), True),
            StructField("identity_hate", IntegerType(), True),
            StructField("input_word_ids", StringType(), True),
            StructField("input_mask", StringType(), True),
            StructField("all_segment_id", StringType(), True)
        ])
        
        try:
            df = self.spark.read.csv(
                file_path,
                header=True,
                schema=schema,
                inferSchema=False,
                mode="DROPMALFORMED"
            )
            
            print(f"[INFO] Successfully loaded {df.count()} records")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to read data: {str(e)}")
            raise
    
    def validate_data(self, df):
        """
        Validate the loaded data
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            Validated DataFrame
        """
        print("[INFO] Validating data...")
        
        # Select only required columns
        required_cols = ["id", "comment_text"] + config.TOXICITY_LABELS
        df = df.select(*required_cols)
        
        # Remove rows with null comment_text
        df = df.filter(col("comment_text").isNotNull())
        
        # Remove empty comments
        df = df.filter(trim(col("comment_text")) != "")
        
        # Remove very short comments (less than 3 characters)
        df = df.filter(length(trim(col("comment_text"))) >= 3)
        
        # Fill null toxicity labels with 0
        for label in config.TOXICITY_LABELS:
            df = df.fillna({label: 0})
        
        # Add text length column
        df = df.withColumn("text_length", length(col("comment_text")))
        
        print(f"[INFO] Validation complete. Records after validation: {df.count()}")
        
        return df
    
    def get_data_statistics(self, df):
        """
        Get basic statistics about the dataset
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            Dictionary with statistics
        """
        print("[INFO] Calculating data statistics...")
        
        total_records = df.count()
        
        stats = {
            "total_records": total_records
        }
        
        # Calculate toxicity distribution
        for label in config.TOXICITY_LABELS:
            toxic_count = df.filter(col(label) == 1).count()
            stats[f"{label}_count"] = toxic_count
            stats[f"{label}_percentage"] = (toxic_count / total_records * 100) if total_records > 0 else 0
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total Records: {stats['total_records']}")
        print(f"\nToxicity Distribution:")
        for label in config.TOXICITY_LABELS:
            print(f"  {label}: {stats[f'{label}_count']} ({stats[f'{label}_percentage']:.2f}%)")
        print(f"{'='*60}\n")
        
        return stats
