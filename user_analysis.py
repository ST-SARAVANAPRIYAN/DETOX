"""
User Toxicity Analysis Module for Detox
Aggregates toxicity metrics at user level
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, avg, max, min, sum as spark_sum,
    when, round as spark_round, collect_list, concat_ws, lit
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import config


class UserToxicityAnalyzer:
    """
    Class to analyze toxicity at user level
    """
    
    def __init__(self):
        pass
    
    def aggregate_user_toxicity(self, predictions_df: DataFrame) -> DataFrame:
        """
        Aggregate toxicity metrics for each user (based on message ID patterns)
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            DataFrame with user-level toxicity metrics
        """
        print("[INFO] Aggregating user-level toxicity metrics...")
        
        # Extract user_id from message id (assuming first 8 characters represent user)
        predictions_df = predictions_df.withColumn(
            "user_id",
            F.substring(col("id"), 1, 8)
        )
        
        # Aggregate metrics per user
        user_aggregates = predictions_df.groupBy("user_id").agg(
            count("id").alias("total_messages"),
            avg("toxicity_score").alias("avg_toxicity_score"),
            max("toxicity_score").alias("max_toxicity_score"),
            min("toxicity_score").alias("min_toxicity_score"),
            spark_sum(when(col("toxicity_level") == "SEVERE", 1).otherwise(0)).alias("severe_toxic_count"),
            spark_sum(when(col("toxicity_level") == "HIGH", 1).otherwise(0)).alias("high_toxic_count"),
            spark_sum(when(col("toxicity_level") == "MODERATE", 1).otherwise(0)).alias("moderate_toxic_count"),
            spark_sum(when(col("toxicity_level") == "LOW", 1).otherwise(0)).alias("low_toxic_count"),
            spark_sum(when(col("prediction") == 1, 1).otherwise(0)).alias("toxic_messages_count"),
            collect_list("id").alias("message_ids")
        )
        
        # Calculate toxicity percentage
        user_aggregates = user_aggregates.withColumn(
            "toxicity_percentage",
            spark_round((col("toxic_messages_count") / col("total_messages")) * 100, 2)
        )
        
        # Round scores
        user_aggregates = user_aggregates.withColumn(
            "avg_toxicity_score",
            spark_round(col("avg_toxicity_score"), 4)
        ).withColumn(
            "max_toxicity_score",
            spark_round(col("max_toxicity_score"), 4)
        ).withColumn(
            "min_toxicity_score",
            spark_round(col("min_toxicity_score"), 4)
        )
        
        # Determine overall user toxicity level
        user_aggregates = user_aggregates.withColumn(
            "user_toxicity_level",
            when(col("avg_toxicity_score") >= config.SEVERE_TOXICITY_THRESHOLD, "VERY_HIGH")
            .when(col("avg_toxicity_score") >= config.HIGH_TOXICITY_THRESHOLD, "HIGH")
            .when(col("avg_toxicity_score") >= config.TOXICITY_THRESHOLD, "MODERATE")
            .when(col("avg_toxicity_score") >= 0.3, "LOW")
            .otherwise("MINIMAL")
        )
        
        # Convert message_ids list to string
        user_aggregates = user_aggregates.withColumn(
            "message_ids_list",
            concat_ws(",", col("message_ids"))
        ).drop("message_ids")
        
        # Sort by toxicity score descending
        user_aggregates = user_aggregates.orderBy(col("avg_toxicity_score").desc())
        
        print(f"[INFO] User aggregation complete. Total users: {user_aggregates.count()}")
        
        return user_aggregates
    
    def get_top_toxic_users(self, user_aggregates: DataFrame, top_n: int = 10) -> DataFrame:
        """
        Get top N most toxic users
        
        Args:
            user_aggregates: DataFrame with user aggregates
            top_n: Number of top users to return
            
        Returns:
            DataFrame with top toxic users
        """
        print(f"[INFO] Getting top {top_n} toxic users...")
        
        top_users = user_aggregates.orderBy(col("avg_toxicity_score").desc()).limit(top_n)
        
        return top_users
    
    def get_user_statistics(self, user_aggregates: DataFrame) -> dict:
        """
        Get overall user toxicity statistics
        
        Args:
            user_aggregates: DataFrame with user aggregates
            
        Returns:
            Dictionary with statistics
        """
        print("[INFO] Calculating user statistics...")
        
        stats = user_aggregates.agg(
            count("user_id").alias("total_users"),
            avg("avg_toxicity_score").alias("overall_avg_toxicity"),
            max("max_toxicity_score").alias("highest_toxicity_score"),
            spark_sum("total_messages").alias("total_messages"),
            spark_sum("toxic_messages_count").alias("total_toxic_messages")
        ).collect()[0]
        
        # Count users by toxicity level
        level_counts = user_aggregates.groupBy("user_toxicity_level").count().collect()
        
        statistics = {
            "total_users": stats["total_users"],
            "overall_avg_toxicity": round(stats["overall_avg_toxicity"], 4),
            "highest_toxicity_score": round(stats["highest_toxicity_score"], 4),
            "total_messages": stats["total_messages"],
            "total_toxic_messages": stats["total_toxic_messages"],
            "toxicity_rate": round((stats["total_toxic_messages"] / stats["total_messages"]) * 100, 2) if stats["total_messages"] > 0 else 0
        }
        
        for level in level_counts:
            statistics[f"users_{level['user_toxicity_level'].lower()}"] = level["count"]
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"USER TOXICITY STATISTICS")
        print(f"{'='*60}")
        print(f"Total Users:                {statistics['total_users']}")
        print(f"Total Messages:             {statistics['total_messages']}")
        print(f"Total Toxic Messages:       {statistics['total_toxic_messages']}")
        print(f"Overall Toxicity Rate:      {statistics['toxicity_rate']:.2f}%")
        print(f"Average Toxicity Score:     {statistics['overall_avg_toxicity']:.4f}")
        print(f"Highest Toxicity Score:     {statistics['highest_toxicity_score']:.4f}")
        print(f"\nUser Distribution by Toxicity Level:")
        for level in ["VERY_HIGH", "HIGH", "MODERATE", "LOW", "MINIMAL"]:
            key = f"users_{level.lower()}"
            if key in statistics:
                print(f"  {level}: {statistics[key]} users")
        print(f"{'='*60}\n")
        
        return statistics
    
    def create_detailed_message_report(self, predictions_df: DataFrame) -> DataFrame:
        """
        Create detailed report with each message and its toxicity metrics
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            DataFrame with detailed message report
        """
        print("[INFO] Creating detailed message report...")
        
        # Extract user_id
        report = predictions_df.withColumn(
            "user_id",
            F.substring(col("id"), 1, 8)
        )
        
        # Select relevant columns
        report = report.select(
            "id",
            "user_id",
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
        )
        
        # Add timestamp (simulated)
        report = report.withColumn(
            "message_index",
            F.monotonically_increasing_id()
        )
        
        print("[INFO] Detailed message report created")
        
        return report
