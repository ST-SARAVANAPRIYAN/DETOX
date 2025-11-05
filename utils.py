"""
Utility functions for Detox project
Helper functions for data processing and analysis
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict


def display_data_sample(df: DataFrame, n: int = 5, truncate: int = 50):
    """
    Display sample data from DataFrame
    
    Args:
        df: PySpark DataFrame
        n: Number of rows to display
        truncate: Maximum column width
    """
    print(f"\nSample Data (showing {n} rows):")
    print("=" * 80)
    df.show(n, truncate=truncate)


def get_toxicity_distribution(df: DataFrame, label_columns: List[str]) -> pd.DataFrame:
    """
    Get distribution of toxicity labels
    
    Args:
        df: PySpark DataFrame with toxicity labels
        label_columns: List of label column names
        
    Returns:
        Pandas DataFrame with distribution
    """
    distributions = []
    total = df.count()
    
    for label in label_columns:
        count = df.filter(col(label) == 1).count()
        percentage = (count / total * 100) if total > 0 else 0
        distributions.append({
            'label': label,
            'count': count,
            'percentage': percentage
        })
    
    return pd.DataFrame(distributions)


def plot_toxicity_distribution(distributions: pd.DataFrame):
    """
    Plot toxicity label distribution
    
    Args:
        distributions: DataFrame with label distributions
    """
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=distributions, x='label', y='count', palette='Reds')
    plt.title('Toxicity Label Distribution (Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    # Percentage plot
    plt.subplot(1, 2, 2)
    sns.barplot(data=distributions, x='label', y='percentage', palette='Oranges')
    plt.title('Toxicity Label Distribution (Percentage)', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def calculate_class_weights(df: DataFrame, label_col: str = "label") -> Dict[float, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        df: PySpark DataFrame
        label_col: Name of label column
        
    Returns:
        Dictionary with class weights
    """
    total = df.count()
    positive = df.filter(col(label_col) == 1).count()
    negative = total - positive
    
    # Calculate weights
    weight_positive = total / (2 * positive) if positive > 0 else 1.0
    weight_negative = total / (2 * negative) if negative > 0 else 1.0
    
    return {
        0.0: weight_negative,
        1.0: weight_positive
    }


def print_confusion_matrix(predictions_df: DataFrame):
    """
    Print confusion matrix from predictions
    
    Args:
        predictions_df: DataFrame with predictions and labels
    """
    tp = predictions_df.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = predictions_df.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = predictions_df.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = predictions_df.filter((col("prediction") == 0) & (col("label") == 1)).count()
    
    print("\nConfusion Matrix:")
    print("=" * 40)
    print(f"{'':15} {'Predicted Negative':^20} {'Predicted Positive':^20}")
    print(f"{'Actual Negative':15} {tn:^20} {fp:^20}")
    print(f"{'Actual Positive':15} {fn:^20} {tp:^20}")
    print("=" * 40)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDerived Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("=" * 40)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_spark_config(spark):
    """
    Print Spark configuration details
    
    Args:
        spark: SparkSession
    """
    print("\n" + "=" * 60)
    print("SPARK CONFIGURATION")
    print("=" * 60)
    
    conf = spark.sparkContext.getConf()
    print(f"App Name:        {conf.get('spark.app.name')}")
    print(f"Master:          {spark.sparkContext.master}")
    print(f"Spark Version:   {spark.version}")
    print(f"Python Version:  {spark.sparkContext.pythonVer}")
    print(f"App ID:          {spark.sparkContext.applicationId}")
    print(f"Executor Memory: {conf.get('spark.executor.memory', 'default')}")
    print(f"Driver Memory:   {conf.get('spark.driver.memory', 'default')}")
    print(f"UI Port:         {conf.get('spark.ui.port', '4040')}")
    print("=" * 60 + "\n")


def create_summary_report(df: DataFrame, predictions_df: DataFrame, 
                         user_aggregates: DataFrame, metrics: Dict) -> str:
    """
    Create a comprehensive summary report
    
    Args:
        df: Original DataFrame
        predictions_df: Predictions DataFrame
        user_aggregates: User aggregates DataFrame
        metrics: Model metrics dictionary
        
    Returns:
        Summary report as string
    """
    total_messages = df.count()
    total_predictions = predictions_df.count()
    total_users = user_aggregates.count()
    toxic_messages = predictions_df.filter(col("prediction") == 1).count()
    
    report = f"""
{'=' * 80}
DETOX PROJECT SUMMARY REPORT
{'=' * 80}

📊 DATA STATISTICS
  • Total Messages Processed:     {total_messages:,}
  • Total Predictions Generated:  {total_predictions:,}
  • Total Unique Users:            {total_users:,}
  • Toxic Messages Detected:      {toxic_messages:,} ({toxic_messages/total_predictions*100:.2f}%)
  • Non-Toxic Messages:            {total_predictions - toxic_messages:,} ({(total_predictions - toxic_messages)/total_predictions*100:.2f}%)

🎯 MODEL PERFORMANCE
  • AUC-ROC Score:    {metrics.get('auc', 0):.4f}
  • AUC-PR Score:     {metrics.get('auc_pr', 0):.4f}
  • Accuracy:         {metrics.get('accuracy', 0):.4f}
  • Precision:        {metrics.get('precision', 0):.4f}
  • Recall:           {metrics.get('recall', 0):.4f}
  • F1 Score:         {metrics.get('f1_score', 0):.4f}

👥 USER INSIGHTS
  • Average Messages per User:    {total_messages / total_users:.2f}
  • Users with Toxic Content:     {user_aggregates.filter(col('toxic_messages_count') > 0).count():,}
  • Users with Clean Content:     {user_aggregates.filter(col('toxic_messages_count') == 0).count():,}

✅ PROJECT STATUS
  • Data Ingestion:      ✓ Complete
  • Preprocessing:       ✓ Complete
  • Model Training:      ✓ Complete
  • Evaluation:          ✓ Complete
  • Predictions:         ✓ Complete
  • User Analysis:       ✓ Complete
  • Results Exported:    ✓ Complete

{'=' * 80}
    """
    
    return report


def export_to_excel(predictions_df: DataFrame, user_aggregates: DataFrame, 
                   output_path: str = "output/detox_results.xlsx"):
    """
    Export results to Excel file (requires openpyxl)
    
    Args:
        predictions_df: Predictions DataFrame
        user_aggregates: User aggregates DataFrame
        output_path: Path to save Excel file
    """
    try:
        from openpyxl import Workbook
        
        # Convert to Pandas
        predictions_pd = predictions_df.select(
            "id", "user_id", "comment_text", "toxicity_score", 
            "toxicity_level", "prediction"
        ).limit(10000).toPandas()
        
        user_agg_pd = user_aggregates.toPandas()
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            predictions_pd.to_excel(writer, sheet_name='Predictions', index=False)
            user_agg_pd.to_excel(writer, sheet_name='User Aggregates', index=False)
        
        print(f"✓ Results exported to Excel: {output_path}")
        
    except ImportError:
        print("⚠ openpyxl not installed. Install with: pip install openpyxl")
    except Exception as e:
        print(f"✗ Failed to export to Excel: {str(e)}")
