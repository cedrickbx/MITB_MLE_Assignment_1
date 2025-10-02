import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    if "attribute" in bronze_directory:
        csv_file_path = "data/features_attributes.csv"
        partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    elif "financials" in bronze_directory:
        csv_file_path = "data/features_financials.csv"
        partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    elif "clickstream" in bronze_directory:
        csv_file_path = "data/feature_clickstream.csv"
        partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    else:
        csv_file_path = "data/lms_loan_daily.csv"
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    filepath = bronze_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
