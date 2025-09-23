#Import modules
import os, glob, random, pprint, pyspark, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
#table processing scripts
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

try:
    print("Initialising Spark session")
    # Initialize SparkSession (cluster)
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session initialised")
except Exception:
    print(f"Error when initializing SparkSession: {Exception}", file=sys.stderr)
    sys.exit(1)

time.sleep(2)
# set up config
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

print("Creating bronze datalake")
# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

time.sleep(1)
# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_attributes_directory, spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_financials_directory, spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_clickstream_directory, spark)

print("Bronze datalake created. Proceeding with silver datalake creation.")
time.sleep(2)
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)
time.sleep(1)
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_financials_directory, silver_financials_directory, spark)
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)

print("Silver datalake created. Proceeding with gold datalake creation.")
time.sleep(2)
# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)
time.sleep(1)
# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(date_str, gold_feature_store_directory, spark, mob = 6)
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

print("Gold datalake created. Creation of datalakes completed.")
time.sleep(2)
print("Showing feature store for inspection.")
folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())
df.show()
time.sleep(2)
print("Showing label store for inspection.")
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())
df.show()
time.sleep(5)



