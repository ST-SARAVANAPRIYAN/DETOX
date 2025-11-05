"""
Data Preprocessing Module for Detox
Handles text cleaning, tokenization, and feature engineering
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, lower, regexp_replace, trim, when
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline
import config
import re


class TextPreprocessor:
    """
    Class to handle text preprocessing operations
    """
    
    def __init__(self):
        self.pipeline = None
        self.pipeline_model = None
    
    @staticmethod
    def clean_text_udf():
        """
        UDF to clean text - remove special characters, URLs, etc.
        """
        def clean(text):
            if text is None:
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespaces
            text = ' '.join(text.split())
            
            return text
        
        return udf(clean, StringType())
    
    def preprocess_text(self, df: DataFrame) -> DataFrame:
        """
        Apply text preprocessing steps
        
        Args:
            df: Input DataFrame with comment_text column
            
        Returns:
            DataFrame with cleaned text
        """
        print("[INFO] Starting text preprocessing...")
        
        # Apply text cleaning
        clean_udf = self.clean_text_udf()
        df = df.withColumn("cleaned_text", clean_udf(col("comment_text")))
        
        # Remove empty cleaned texts
        df = df.filter(trim(col("cleaned_text")) != "")
        
        print(f"[INFO] Text preprocessing complete. Records: {df.count()}")
        
        return df
    
    def build_feature_pipeline(self):
        """
        Build ML pipeline for feature extraction
        
        Returns:
            PySpark ML Pipeline
        """
        print("[INFO] Building feature extraction pipeline...")
        
        # Tokenization
        tokenizer = Tokenizer(
            inputCol="cleaned_text",
            outputCol="words"
        )
        
        # Remove stop words
        stop_words_remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        
        # Term Frequency (TF)
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=config.MAX_FEATURES
        )
        
        # Inverse Document Frequency (IDF)
        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
            minDocFreq=config.IDF_MIN_DOC_FREQ
        )
        
        # Create pipeline
        self.pipeline = Pipeline(stages=[
            tokenizer,
            stop_words_remover,
            hashing_tf,
            idf
        ])
        
        print("[INFO] Feature extraction pipeline built successfully")
        
        return self.pipeline
    
    def fit_transform_features(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with features
        """
        print("[INFO] Fitting and transforming features...")
        
        if self.pipeline is None:
            self.build_feature_pipeline()
        
        self.pipeline_model = self.pipeline.fit(df)
        df_features = self.pipeline_model.transform(df)
        
        print("[INFO] Feature extraction complete")
        
        return df_features
    
    def transform_features(self, df: DataFrame) -> DataFrame:
        """
        Transform features using fitted pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with features
        """
        if self.pipeline_model is None:
            raise ValueError("Pipeline not fitted yet. Call fit_transform_features first.")
        
        return self.pipeline_model.transform(df)
    
    def create_label_column(self, df: DataFrame, label_col: str = "toxic") -> DataFrame:
        """
        Create a single label column for classification
        
        Args:
            df: Input DataFrame
            label_col: Name of the toxicity label column to use
            
        Returns:
            DataFrame with label column
        """
        print(f"[INFO] Creating label column from '{label_col}'...")
        
        # Ensure the label is properly cast to integer first, then to double
        # Also handle any null or invalid values
        df = df.withColumn(
            "label", 
            when(col(label_col).isNull(), 0.0)
            .when(col(label_col) == 1, 1.0)
            .otherwise(0.0)
        )
        
        return df
