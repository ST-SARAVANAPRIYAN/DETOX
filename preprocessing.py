"""
Data Preprocessing Module for Detox
Handles text cleaning, tokenization, and feature engineering
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, lower, regexp_replace, trim, when
from pyspark.sql.types import StringType, ArrayType, IntegerType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline
import config
import re
from toxicity_lexicon import get_toxic_word_count


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
    
    @staticmethod
    def add_lexicon_features(df: DataFrame) -> DataFrame:
        """
        Add toxicity lexicon-based features to DataFrame
        
        Args:
            df: DataFrame with cleaned_text column
            
        Returns:
            DataFrame with additional lexicon features
        """
        print("[INFO] Adding toxicity lexicon features...")
        
        # UDF to extract lexicon features
        def extract_lexicon_features(text):
            if text is None or text == "":
                return (0, 0, 0, 0, 0, 0.0, 0.0)
            
            stats = get_toxic_word_count(text)
            return (
                stats['extreme_toxic_count'],
                stats['high_toxic_count'],
                stats['medium_toxic_count'],
                stats['low_toxic_count'],
                stats['total_toxic_count'],
                float(stats['toxic_word_ratio']),
                float(stats['severity_score'])
            )
        
        from pyspark.sql.types import StructType, StructField
        
        lexicon_schema = StructType([
            StructField("extreme_toxic_count", IntegerType(), False),
            StructField("high_toxic_count", IntegerType(), False),
            StructField("medium_toxic_count", IntegerType(), False),
            StructField("low_toxic_count", IntegerType(), False),
            StructField("total_toxic_count", IntegerType(), False),
            StructField("toxic_word_ratio", FloatType(), False),
            StructField("severity_score", FloatType(), False)
        ])
        
        lexicon_udf = udf(extract_lexicon_features, lexicon_schema)
        
        # Add lexicon features
        df = df.withColumn("lexicon_features", lexicon_udf(col("cleaned_text")))
        
        # Expand struct into separate columns
        df = df.withColumn("extreme_toxic_count", col("lexicon_features.extreme_toxic_count")) \
               .withColumn("high_toxic_count", col("lexicon_features.high_toxic_count")) \
               .withColumn("medium_toxic_count", col("lexicon_features.medium_toxic_count")) \
               .withColumn("low_toxic_count", col("lexicon_features.low_toxic_count")) \
               .withColumn("total_toxic_count", col("lexicon_features.total_toxic_count")) \
               .withColumn("toxic_word_ratio", col("lexicon_features.toxic_word_ratio")) \
               .withColumn("severity_score", col("lexicon_features.severity_score")) \
               .drop("lexicon_features")
        
        print("[INFO] Lexicon features added successfully")
        print(f"[INFO] Sample severity scores: {df.select('severity_score').limit(5).toPandas()['severity_score'].tolist()}")
        
        return df
    
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
            outputCol="tfidf_features",  # Changed from "features"
            minDocFreq=config.IDF_MIN_DOC_FREQ
        )
        
        # Combine TF-IDF features with lexicon features
        from pyspark.ml.feature import VectorAssembler
        
        feature_assembler = VectorAssembler(
            inputCols=[
                "tfidf_features",
                "extreme_toxic_count",
                "high_toxic_count", 
                "medium_toxic_count",
                "low_toxic_count",
                "total_toxic_count",
                "toxic_word_ratio",
                "severity_score"
            ],
            outputCol="features"
        )
        
        # Create pipeline with lexicon features
        self.pipeline = Pipeline(stages=[
            tokenizer,
            stop_words_remover,
            hashing_tf,
            idf,
            feature_assembler
        ])
        
        print("[INFO] Feature extraction pipeline built successfully (TF-IDF + Lexicon)")

        
        return self.pipeline
    
    def fit_transform_features(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform features (TF-IDF + Lexicon)
        
        Args:
            df: Input DataFrame with cleaned_text column
            
        Returns:
            DataFrame with combined features
        """
        print("[INFO] Adding lexicon features before pipeline...")
        
        # Add lexicon features first
        df = self.add_lexicon_features(df)
        
        print("[INFO] Fitting and transforming TF-IDF + Lexicon features...")
        
        if self.pipeline is None:
            self.build_feature_pipeline()
        
        self.pipeline_model = self.pipeline.fit(df)
        df_features = self.pipeline_model.transform(df)
        
        print("[INFO] Feature extraction complete (TF-IDF + 7 lexicon features)")
        
        return df_features
    
    def transform_features(self, df: DataFrame) -> DataFrame:
        """
        Transform features using fitted pipeline (includes lexicon features)
        
        Args:
            df: Input DataFrame with cleaned_text column
            
        Returns:
            DataFrame with features
        """
        if self.pipeline_model is None:
            raise ValueError("Pipeline not fitted yet. Call fit_transform_features first.")
        
        # Add lexicon features first
        df = self.add_lexicon_features(df)
        
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
