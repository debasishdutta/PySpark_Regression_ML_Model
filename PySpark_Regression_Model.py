#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Downloads/Python Code Library/PySpark Model Codes/Datasets/Regression_Model_Dataset.csv" 
global_source_format = "csv"
global_dep_var = 'AvgBill'
global_id_var = 'id'
global_train_split = 0.8
global_seed =1234

# Model Configurations (Linear Regression, Generalized Linear Model, Random Forest, Gradient Boosting)
model_param_max_iter = 100
model_param_max_depth = 10
model_param_max_bins = 5
model_param_n_trees = 1000
model_param_fit_intercept = True
model_param_lr_standardize = False
model_param_elasticnet_param = 0.8
model_param_reg_param = 0.4


# In[ ]:


### ENVIORNMENT SET UP ###

# Initialize PySpark Engine #
import findspark
findspark.init()

# Initiate A Spark Session On Local Machine With 4 Physical Cores #
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ML_Regression_Pyspark_V1').master('local[4]').getOrCreate()


# In[ ]:


### RAW DATA IMPORT ###

from pyspark.sql.types import *
from time import *

def data_import(source_name, source_format):
    
    import_start_time = time()
    
    print("\nSpark Session Initiated Successfully. Kindly Follow The Log For Further Output\n")
    
    df = spark.read.format(source_format).option("header","true").option("inferSchema","true").load(source_name)
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)/60
    print("\nTime To Perform Data Import: %.3f Minutes\n" % import_elapsed_time)
    
    return(df)


# In[ ]:


### USER DEFINED FUNCTION: EXPLORATORY DATA ANALYSIS (EDA) ###

from time import *
import pandas as pd
from pyspark.sql.functions import isnan, when, count, col
  
def basic_eda(df,dependent_var,id_var):
    
    eda_start_time = time()
  
  # Extracting Data Types of All Columns
    print("\n++++++ Printing Data Types of All Columns ++++++\n")
    df.printSchema()
  
  # Duplicate Observation Checking
    print("\n++++++ Printing Duplicate Removal Summary ++++++\n")
    print("Total No of Obs Before Duplicate Removal: "+str(df.count()))
    print("Unique No of Obs Before Duplicate Removal: "+str(df.distinct().count()))
  
  # Removing Duplicate Observations
    df = df.dropDuplicates()
    df = df.na.drop('all')
    print("Total No of Obs After Duplicate Removal: "+str(df.count()))
    print("Unique No of Obs After Duplicate Removal: "+str(df.distinct().count()))
  
  # Extracting Dependent and Independent Variables
    column_names = [item[0] for item in df.dtypes]
    categorical_var = [item[0] for item in df.dtypes if item[1].startswith('string')]
    independent_catgorical_var=[x for x in categorical_var if x not in [id_var,dependent_var]]
    independent_continuous_var=[x for x in column_names if x not in independent_catgorical_var+[id_var,dependent_var]]
 
  # Descriptive Summary of Numeric Variables
    temp_df_1 = pd.DataFrame()
    desc_summary_1 = pd.DataFrame()
    
    for col_name in df[independent_continuous_var].columns:
        temp_df_1.loc[0,"Column_Name"] = col_name
        temp_df_1.loc[0,"Total_Obs"] = df.agg({col_name: "count"}).collect()[0][0]
        temp_df_1.loc[0,"Unique_No_Obs"] = df.select(col_name).distinct().count()
        temp_df_1.loc[0,"Missing_No_Obs"] = df.select(count(when(isnan(col_name)
                                                             |col(col_name).isNull(), col_name))).toPandas().iloc[0,0]
        temp_df_1.loc[0,"Min"] = df.agg({col_name: "min"}).collect()[0][0]
        temp_var = df.approxQuantile(col_name,[0.01,0.05,0.1,0.25,0.5,0.75,0.85,0.95,0.99,],0)
        temp_df_1.loc[0,"Pct_1"] = temp_var[0]
        temp_df_1.loc[0,"Pct_5"] = temp_var[1]
        temp_df_1.loc[0,"Pct_10"] = temp_var[2]
        temp_df_1.loc[0,"Pct_25"] = temp_var[3]
        temp_df_1.loc[0,"Median"] = temp_var[4]
        temp_df_1.loc[0,"Average"] = df.agg({col_name: "avg"}).collect()[0][0]
        temp_df_1.loc[0,"Pct_75"] = temp_var[5]
        temp_df_1.loc[0,"Pct_85"] = temp_var[6]
        temp_df_1.loc[0,"Pct_95"] = temp_var[7]
        temp_df_1.loc[0,"Pct_99"] = temp_var[8]
        temp_df_1.loc[0,"Max"] = df.agg({col_name: "max"}).collect()[0][0]
        desc_summary_1 = desc_summary_1.append(temp_df_1)
        desc_summary_1.reset_index(inplace = True, drop = True)       

    print("\n++++++ Printing Summary Statistics For Numeric Variables ++++++\n")
    display(desc_summary_1)
    
    # Target Variables V/s Categorical Variables
    temp_df_2 = pd.DataFrame()
    desc_summary_2 = pd.DataFrame()
    
    for x in independent_catgorical_var:
        temp_df_2 = df.groupby(x).agg({dependent_var: "avg"}).toPandas()
        temp_df_2.columns = ["Column_Value","Avg_Target_Var"]
        temp_df_2["Column_Name"] = x
        temp_df_2 = temp_df_2.iloc[:,[2,0,1]]
        desc_summary_2 = desc_summary_2.append(temp_df_2)

    print("\n++++++ Printing Averages of Target Variable Grouped By All Categorical Variable ++++++\n")
    display(desc_summary_2)
    
  # Returning Final Output
    desc_summary = [desc_summary_1,desc_summary_2]
    final_list = (df,independent_catgorical_var,independent_continuous_var, desc_summary)
    
    eda_end_time = time()
    eda_elapsed_time = (eda_end_time - eda_start_time)/60
    print("\nTime To Perform EDA: %.3f Minutes\n" % eda_elapsed_time)
  
    return(final_list)


# In[ ]:


### USER DEFINED FUNCTION: FEATURE ENGINEERING ###

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from time import *

def feature_engineering(df,independent_catgorical_var,independent_continuous_var,dependent_var,id_var):
    
    fe_start_time = time()
  
  # Initiating pipeline 
    stages = []
    for categoricalCol in independent_catgorical_var:
        # Convert Categorical Variables In To Numeric Indices
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + '_Index')
        # Perform One Hot Encoding
        onehotEncoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_classVec"])
        stages += [stringIndexer, onehotEncoder]

  # Index The Target Variable
    label_stringIdx = StringIndexer(inputCol = dependent_var, outputCol = 'label')
    stages += [label_stringIdx]
    
  # Assembling All Features
    assemblerInputs = [c + "_classVec" for c in independent_catgorical_var] + independent_continuous_var

  # Creating Feature Vector
    vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [vecAssembler]

  # Finalizing Pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df_transformed = pipelineModel.transform(df)
    selectedCols = [id_var, 'label', 'features'] 
    final_df = df_transformed.select(selectedCols)
    print("\n++++++ Printing Structure of Final Dataset ++++++\n")
    final_df.printSchema()
    
    fe_end_time = time()
    fe_elapsed_time = (fe_end_time - fe_start_time)/60
    print("\nTime To Perform Feature Engineering: %.3f Minutes\n" % fe_elapsed_time)
    
    return(final_df)


# In[ ]:


### USER DEFINED FUNCTION: TRAIN & TEST SAMPLE CREATION USING RANDOM SAMPLING ###

from time import *

def random_sampling(final_df,train_prop, seed):
  
    sampling_start_time = time()
    
    print("\n++++++ Printing Development & Validation Sample Details ++++++\n")
    train, test = final_df.randomSplit([train_prop, 1-train_prop], seed = seed)
    
    print("Training Dataset Count: " + str(train.count()))
   
    print("Test Dataset Count: " + str(test.count()))
    
    final_list = (train,test)
    
    sampling_end_time = time()
    sampling_elapsed_time = (sampling_end_time - sampling_start_time)/60
    print("\nTime To Perform Data Split: %.3f Minutes\n" % sampling_elapsed_time)  
    
    return (final_list)


# In[ ]:


### USER DEFINED FUNCTION: LINEAR REGRESSION MODEL ###

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from time import *

def model_dev_mlr(df_train, df_test, elasticnet_param, fit_intercept, max_iter, reg_param, lr_standardize):
    
    mlr_start_time = time()
    
    # Create an Initial Model Instance
    mod_mlr = LinearRegression(labelCol='label',
                              featuresCol='features',
                              elasticNetParam=elasticnet_param,
                              fitIntercept=fit_intercept,
                              maxIter=max_iter,
                              regParam=reg_param,
                              standardization=lr_standardize)
    
    # Training The Model
    mlr_final_model = mod_mlr.fit(df_train) 
    
    # Scoring The Model On Test Sample
    mlr_transformed = mlr_final_model.transform(df_test)
    mlr_test_results = mlr_transformed.select(['prediction', 'label'])

    # Collecting The Model Statistics
    mlr_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="label")
    mlr_r2 = round(mlr_evaluator.evaluate(mlr_test_results,{mlr_evaluator.metricName: "r2"}),3)
    mlr_mse = round(mlr_evaluator.evaluate(mlr_test_results,{mlr_evaluator.metricName: "mse"}),3)
    mlr_rmse = round(mlr_evaluator.evaluate(mlr_test_results,{mlr_evaluator.metricName: "rmse"}),3)
    mlr_mae = round(mlr_evaluator.evaluate(mlr_test_results,{mlr_evaluator.metricName: "mae"}),3)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Linear Regression Model Accuracy ++++++\n")
    print("R Square: "+str(mlr_r2*100)+"%")
    print("Mean Squared Error: "+str(mlr_mse))
    print("Root Mean Squared Error: "+str(mlr_rmse))
    print("Mean Absolute Error: "+str(mlr_mae))

    mlr_end_time = time()
    mlr_elapsed_time = (mlr_end_time - mlr_start_time)/60
    mlr_model_stat = pd.DataFrame({"Model Name" : ["Linear Regression"],
                                  "R Square" : mlr_r2,
                                  "Mean Squared Error": mlr_mse, 
                                  "Root Mean Squared Error": mlr_rmse,
                                  "Mean Absolute Error": mlr_mae, 
                                  "Time (Min.)": round(mlr_elapsed_time,3)})
    mlr_output = (mlr_final_model,mlr_model_stat)
    
    return(mlr_output)


# In[ ]:


### USER DEFINED FUNCTION: GENERALIZED LINEAR MODEL ###

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from time import *

def model_dev_glm(df_train, df_test, max_iter, fit_intercept, reg_param):
    
    glm_start_time = time()
    
    # Create an Initial Model Instance
    mod_glm = GeneralizedLinearRegression(labelCol='label',
                                          featuresCol='features',
                                          family="gaussian",
                                          link="identity",                                          
                                          fitIntercept=fit_intercept,
                                          maxIter=max_iter,
                                          regParam=reg_param)
    
    # Training The Model
    glm_final_model = mod_glm.fit(df_train) 
    
    # Scoring The Model On Test Sample
    glm_transformed = glm_final_model.transform(df_test)
    glm_test_results = glm_transformed.select(['prediction', 'label'])

    # Collecting The Model Statistics
    glm_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="label")
    glm_r2 = round(glm_evaluator.evaluate(glm_test_results,{glm_evaluator.metricName: "r2"}),3)
    glm_mse = round(glm_evaluator.evaluate(glm_test_results,{glm_evaluator.metricName: "mse"}),3)
    glm_rmse = round(glm_evaluator.evaluate(glm_test_results,{glm_evaluator.metricName: "rmse"}),3)
    glm_mae = round(glm_evaluator.evaluate(glm_test_results,{glm_evaluator.metricName: "mae"}),3)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Generalized Linear Model Accuracy ++++++\n")
    print("R Square: "+str(glm_r2*100)+"%")
    print("Mean Squared Error: "+str(glm_mse))
    print("Root Mean Squared Error: "+str(glm_rmse))
    print("Mean Absolute Error: "+str(glm_mae))

    glm_end_time = time()
    glm_elapsed_time = (glm_end_time - glm_start_time)/60
    glm_model_stat = pd.DataFrame({"Model Name" : ["Generalized Linear Model"],
                                   "R Square" : glm_r2,
                                   "Mean Squared Error": glm_mse, 
                                   "Root Mean Squared Error": glm_rmse,
                                   "Mean Absolute Error": glm_mae, 
                                   "Time (Min.)": round(glm_elapsed_time,3)})
    glm_output = (glm_final_model,glm_model_stat)
    
    return(glm_output)


# In[ ]:


### USER DEFINED FUNCTION: RANDOM FOREST MODEL ###

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from time import *

def model_dev_rf(df_train, df_test, n_trees, max_bins, max_depth):
    
    rf_start_time = time()
    
    # Create an Initial Model Instance
    mod_rf = RandomForestRegressor(labelCol='label',
                                   featuresCol='features',
                                   impurity='variance',
                                   featureSubsetStrategy='all',
                                   numTrees=n_trees,
                                   maxBins=max_bins,
                                   maxDepth=max_depth)
    
    # Training The Model
    rf_final_model = mod_rf.fit(df_train) 
    
    # Scoring The Model On Test Sample
    rf_transformed = rf_final_model.transform(df_test)
    rf_test_results = rf_transformed.select(['prediction', 'label'])

    # Collecting The Model Statistics
    rf_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="label")
    rf_r2 = round(rf_evaluator.evaluate(rf_test_results,{rf_evaluator.metricName: "r2"}),3)
    rf_mse = round(rf_evaluator.evaluate(rf_test_results,{rf_evaluator.metricName: "mse"}),3)
    rf_rmse = round(rf_evaluator.evaluate(rf_test_results,{rf_evaluator.metricName: "rmse"}),3)
    rf_mae = round(rf_evaluator.evaluate(rf_test_results,{rf_evaluator.metricName: "mae"}),3)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Random Forest Model Accuracy ++++++\n")
    print("R Square: "+str(rf_r2*100)+"%")
    print("Mean Squared Error: "+str(rf_mse))
    print("Root Mean Squared Error: "+str(rf_rmse))
    print("Mean Absolute Error: "+str(rf_mae))

    rf_end_time = time()
    rf_elapsed_time = (rf_end_time - rf_start_time)/60
    rf_model_stat = pd.DataFrame({"Model Name" : ["Random Forest"],
                                  "R Square" : rf_r2,
                                  "Mean Squared Error": rf_mse, 
                                  "Root Mean Squared Error": rf_rmse,
                                  "Mean Absolute Error": rf_mae, 
                                  "Time (Min.)": round(rf_elapsed_time,3)})
    rf_output = (rf_final_model,rf_model_stat)
    
    return(rf_output)


# In[ ]:


### USER DEFINED FUNCTION: GRADIENT BOOSTING MODEL ###

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from time import *

def model_dev_gbm(df_train, df_test, max_iter, max_bins, max_depth):
    
    gbm_start_time = time()
    
    # Create an Initial Model Instance
    mod_gbm = GBTRegressor(labelCol='label',
                           featuresCol='features',
                           featureSubsetStrategy='all',
                           lossType='squared',
                           maxIter=max_iter,
                           maxBins=max_bins,
                           maxDepth=max_depth)
    
    # Training The Model
    gbm_final_model = mod_gbm.fit(df_train) 
    
    # Scoring The Model On Test Sample
    gbm_transformed = gbm_final_model.transform(df_test)
    gbm_test_results = gbm_transformed.select(['prediction', 'label'])

    # Collecting The Model Statistics
    gbm_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="label")
    gbm_r2 = round(gbm_evaluator.evaluate(gbm_test_results,{gbm_evaluator.metricName: "r2"}),3)
    gbm_mse = round(gbm_evaluator.evaluate(gbm_test_results,{gbm_evaluator.metricName: "mse"}),3)
    gbm_rmse = round(gbm_evaluator.evaluate(gbm_test_results,{gbm_evaluator.metricName: "rmse"}),3)
    gbm_mae = round(gbm_evaluator.evaluate(gbm_test_results,{gbm_evaluator.metricName: "mae"}),3)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Gradient Boosting Model Accuracy ++++++\n")
    print("R Square: "+str(gbm_r2*100)+"%")
    print("Mean Squared Error: "+str(gbm_mse))
    print("Root Mean Squared Error: "+str(gbm_rmse))
    print("Mean Absolute Error: "+str(gbm_mae))

    gbm_end_time = time()
    gbm_elapsed_time = (gbm_end_time - gbm_start_time)/60
    gbm_model_stat = pd.DataFrame({"Model Name" : ["Gradient Boosting"],
                                   "R Square" : gbm_r2,
                                   "Mean Squared Error": gbm_mse, 
                                   "Root Mean Squared Error": gbm_rmse,
                                   "Mean Absolute Error": gbm_mae, 
                                   "Time (Min.)": round(gbm_elapsed_time,3)})
    gbm_output = (gbm_final_model,gbm_model_stat)
    
    return(gbm_output)


# In[ ]:


### SCRIPT EXECUTION ###

# Data Import #
raw_data = data_import(global_source_name, global_source_format)

# Exploratory Data Analysis #
eda_output = basic_eda(raw_data,global_dep_var,global_id_var)

# Feature Engineering #
transformed_data = feature_engineering(eda_output[0],eda_output[1],eda_output[2],global_dep_var,global_id_var)

# Random Sampling of Test & Train Data #
sampling  = random_sampling(transformed_data,global_train_split,global_seed)

# Multiple Linear Regression Model #
model_linear_regression = model_dev_mlr(sampling[0],
                                        sampling[1],
                                        model_param_elasticnet_param,
                                        model_param_fit_intercept,
                                        model_param_max_iter,
                                        model_param_reg_param,
                                        model_param_lr_standardize)

# Generalized Linear Model #
model_generalized_linear_regression = model_dev_glm(sampling[0],
                                                    sampling[1],
                                                    model_param_max_iter,
                                                    model_param_fit_intercept,
                                                    model_param_reg_param)

# Random Forest Model #
model_random_forest = model_dev_rf(sampling[0],
                                    sampling[1],
                                    model_param_n_trees,
                                    model_param_max_bins,
                                    model_param_max_depth)

# Gradient Boosting Model #
model_gradient_boosting = model_dev_gbm(sampling[0],
                                        sampling[1],
                                        model_param_max_iter,
                                        model_param_max_bins,
                                        model_param_max_depth)

# Collecting All Model Output #
print("\n++++++ Overall Model Summary ++++++\n")
all_model_summary = pd.DataFrame()
all_model_summary = all_model_summary.append(model_linear_regression[1],ignore_index=True).append(model_generalized_linear_regression[1],ignore_index=True).append(model_random_forest[1],ignore_index=True).append(model_gradient_boosting[1],ignore_index=True)
display(all_model_summary)

print("\n++++++ Process Completed ++++++\n")


# In[ ]:




