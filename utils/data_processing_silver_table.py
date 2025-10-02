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


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")      
        
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    if "attributes" in bronze_directory: #map for attributes features
        # connect to bronze table
        partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = bronze_directory + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())
        df = df.distinct()
        reg_pattern = r"(^_+$)"
        df = df.withColumn("Occupation_invalid", F.trim("Occupation").rlike(reg_pattern)).withColumn("Occupation", F.regexp_replace(F.trim("Occupation"),reg_pattern,"Not_Indicated"))
        int_col_to_process = ["Age"]
        for col_name in int_col_to_process:
            df = df.withColumn(col_name, F.regexp_replace(col_name, "_", "")) #remove invalid values with lagging underscore e.g. 0_
            if col_name == "Age":
                df = df.withColumn(col_name,F.col(col_name).cast("int")).withColumn(col_name+"_out_of_range", ~F.col(col_name).between(0,120))
                df = df.withColumn(col_name+"_clean", F.when(F.col(col_name+"_out_of_range"), F.lit(None)).otherwise(F.col(col_name)))

        column_type_map = {'Customer_ID':StringType(),
                            'Name':StringType(),
                            'Age':IntegerType(),
                            'SSN':StringType(),
                            'Occupation':StringType(),
                            'snapshot_date':DateType()}
        new_partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    elif "financials" in bronze_directory: #map for financial features
        # connect to bronze table
        partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = bronze_directory + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())
        df = df.distinct()
        # col_check_missing = ["Type_of_Loan"]
        # for col_name in df.columns:
        #     if col_name in col_check_missing:
        #         df = df.withColumn(col_name+"_missing", F.col(col_name).isNull() | (F.length(F.trim(col_name)) == 0))
        col_to_process = ["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt", "Credit_Mix", "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", "Monthly_Balance"]
        for col_name in col_to_process:
            if  col_name == "Credit_Mix":
                df = df.withColumn(col_name, F.regexp_replace(F.trim(col_name),r"^_+$","Not_Indicated"))
            else:
                df = df.withColumn(col_name, F.regexp_replace(col_name, "_", "")) #remove invalid values with lagging underscore e.g. 0_
                if col_name in ["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt", "Total_EMI_per_month", "Monthly_Balance"]:
                    df = df.withColumn(col_name,F.col(col_name).cast("int")).withColumn(col_name+"_negative", F.col(col_name) < 0)
                    df = df.withColumn(col_name+"_clean", F.when(F.col(col_name+"_negative"), F.lit(None)).otherwise(F.col(col_name)))                    
        #convert credit history age to float
        pattern = r"(\d+) years and (\d+) months"
        years = F.regexp_extract(F.lower(F.col("Credit_History_Age")), pattern, 1)
        years = F.when(years != "", years.cast("int")).otherwise(F.lit(0))
        months = F.regexp_extract(F.lower(F.col("Credit_History_Age")), pattern, 2)
        months = F.when(months != "", months.cast("int")).otherwise(F.lit(0))
        df = df.withColumn("Credit_History_Age", years+(months/12))
        column_type_map = {'Customer_ID': StringType(),
                                'Annual_Income': FloatType(),
                                'Monthly_Inhand_Salary': FloatType(),
                                'Num_Bank_Accounts': IntegerType(),
                                'Num_Credit_Card': IntegerType(),
                                'Interest_Rate': FloatType(),
                                'Num_of_Loan': IntegerType(),
                                'Type_of_Loan': StringType(),
                                'Delay_from_due_date': IntegerType(),
                                'Num_of_Delayed_Payment': IntegerType(),
                                'Changed_Credit_Limit': FloatType(),
                                'Num_Credit_Inquiries': IntegerType(),
                                'Credit_Mix': StringType(),
                                'Outstanding_Debt': FloatType(),
                                'Credit_Utilization_Ratio': FloatType(),
                                'Credit_History_Age': FloatType(),
                                'Payment_of_Min_Amount': StringType(),
                                'Total_EMI_per_month': FloatType(),
                                'Amount_invested_monthly': FloatType(),
                                'Payment_Behaviour': StringType(),
                                'Monthly_Balance': FloatType(),
                                'snapshot_date': DateType()}
        new_partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    elif "clickstream" in bronze_directory:#map for clickstream features
        # connect to bronze table
        partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = bronze_directory + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())
        df = df.distinct()
        column_type_map = {'fe_1': IntegerType(),
                                'fe_2': IntegerType(),
                                'fe_3': IntegerType(),
                                'fe_4': IntegerType(),
                                'fe_5': IntegerType(),
                                'fe_6': IntegerType(),
                                'fe_7': IntegerType(),
                                'fe_8': IntegerType(),
                                'fe_9': IntegerType(),
                                'fe_10': IntegerType(),
                                'fe_11': IntegerType(),
                                'fe_12': IntegerType(),
                                'fe_13': IntegerType(),
                                'fe_14': IntegerType(),
                                'fe_15': IntegerType(),
                                'fe_16': IntegerType(),
                                'fe_17': IntegerType(),
                                'fe_18': IntegerType(),
                                'fe_19': IntegerType(),
                                'fe_20': IntegerType(),
                                'Customer_ID': StringType(),
                                'snapshot_date': DateType()}
        new_partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    else:
        # connect to bronze table
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = bronze_directory + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())
        df = df.distinct()
        #mapping of dtypes obtained from `lab_2`
        column_type_map = {
            "loan_id": StringType(),
            "Customer_ID": StringType(),
            "loan_start_date": DateType(),
            "tenure": IntegerType(),
            "installment_num": IntegerType(),
            "loan_amt": FloatType(),
            "due_amt": FloatType(),
            "paid_amt": FloatType(),
            "overdue_amt": FloatType(),
            "balance": FloatType(),
            "snapshot_date": DateType(),
        }
        new_partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'

    #casting datatypes
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    #data augmentation depending on the dataset
    if "lms" in bronze_directory:
        # augment data: add month on book
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
        # augment data: add days past due
        df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
        df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
        df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
        
    # save silver table - IRL connect to database to write
    new_filepath = silver_directory + new_partition_name
    df.write.mode("overwrite").parquet(new_filepath)
    # df.toPandas().to_parquet(new_filepath,
    #           compression='gzip')
    print('saved to:', new_filepath)
        
    return df