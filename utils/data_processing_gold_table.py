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

def process_features_gold_table(snapshot_date_str, gold_store_directory, spark, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    #get cust_id of mob to get features of corresponding labels for training
    label_partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    label_filepath = "datamart/silver/loan_daily/" + label_partition_name
    label_df = spark.read.parquet(label_filepath)
    print('loaded from: ', label_filepath, 'row count: ', label_df.count())
    label_df = label_df.filter(col("mob") == mob).select("Customer_ID","snapshot_date").distinct()

    # connect to silver tables of features
    clickstream_partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet' #incremental table - same customer_IDs in every snapshot_date
    clickstream_filepath = "datamart/silver/clickstream/" + clickstream_partition_name
    clickstream_df = spark.read.parquet(clickstream_filepath)
    print('loaded from: ', clickstream_filepath, 'row count: ', clickstream_df.count())
    clickstream_df = clickstream_df.drop("snapshot_date")
    filtered_df = label_df.join(clickstream_df, "Customer_ID", how="left")

    #overwrite tables
    folder_path = "datamart/silver/attributes/"
    attr_files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    attr_df = spark.read.option("header", "true").parquet(*attr_files_list)
    print('loaded: attributes_df')
    attr_df = attr_df.select("Customer_ID", "Age_clean", "Age_out_of_range", "Occupation", "Occupation_invalid")

    folder_path = "datamart/silver/financials/"
    fin_files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    financials_df = spark.read.option("header", "true").parquet(*fin_files_list)
    print('loaded: financials_df')
    financials_df = financials_df.withColumn("Credit_Age", F.when(col("Credit_History_Age") <= 10, 0).otherwise(1)) # 0 stands for 10 years or below credit age
    financials_df = financials_df.withColumn("Credit_Age_def", F.lit("threshold_10_years").cast(StringType())) # 0 stands for 10 years or below credit age
    financials_df = financials_df.select("Customer_ID","Annual_Income_clean", "Monthly_Inhand_Salary_clean", "Num_Bank_Accounts_clean", "Num_Credit_Card_clean", "Interest_Rate_clean", "Num_of_Loan_clean", "Num_of_Delayed_Payment_clean", "Outstanding_Debt_clean", "Credit_Mix", "Credit_Utilization_Ratio", "Total_EMI_per_month_clean", "Monthly_Balance_clean", "Annual_Income_negative", "Monthly_Inhand_Salary_negative", "Num_Bank_Accounts_negative", "Num_Credit_Card_negative", "Interest_Rate_negative", "Num_of_Loan_negative", "Num_of_Delayed_Payment_negative", "Outstanding_Debt_negative", "Total_EMI_per_month_negative", "Monthly_Balance_negative", "Credit_Age","Credit_Age_def")
                                                         
    #merge features df into one feature gold table
    df = filtered_df.join(attr_df, "Customer_ID", "left").join(financials_df, "Customer_ID", "left")

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_labels_gold_table(snapshot_date_str, silver_directory, gold_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType())) #help to define in the dataset what the label is

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df