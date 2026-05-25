import findspark
from pyspark.sql import SparkSession
import logging
from . import config

def init_spark():
    """Initializes and returns a SparkSession."""
    try:
        findspark.init()
    except Exception as e:
        # findspark might fail if SPARK_HOME is not set, but if pyspark is installed via pip,
        # SparkSession.builder might just work natively.
        logging.warning(f"findspark initialization failed: {e}. Attempting direct initialization.")

    spark = SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master(config.SPARK_MASTER) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    return spark
