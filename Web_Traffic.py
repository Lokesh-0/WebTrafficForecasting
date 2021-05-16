# Databricks notebook source
'''
Libraries
'''

import pyspark
from pyspark.sql.functions import isnan, when, count, col
import re
from collections import Counter
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels as sm
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

'''
Spark Session
'''
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark = SparkSession.builder.appName('Bigdata').getOrCreate()

# COMMAND ----------

# MAGIC %fs cp dbfs:/FileStore/tables/train_1.csv file:/FileStore/tables/train_1.csv

# COMMAND ----------

'''
Importing the Dataset
'''
train_1 = "train_1.csv"
df_train = spark.read.format("csv").option("inferSchema","true").option("header","true").load(f"/FileStore/tables/{train_1}")

# COMMAND ----------

df_train.display()

# COMMAND ----------

'''
Total Number of Features and Rows
'''
print(f"Number of rows: {df_train.count()}") 
print(f"Number of columns: {len(df_train.columns)}")

# COMMAND ----------

'''
Data types of the features 
'''
df_train.printSchema()

# COMMAND ----------

for i in df_train.head(5):
  print(i)
  print('\n')

# COMMAND ----------

'''
Counting the Number of Null values present in the dataset
'''
df_train.select([count(when(col(null).isNull(), null)).alias(null) for null in df_train.columns]).display()

# COMMAND ----------

df_train

# COMMAND ----------

'''
Replacing the null values with Zero (0)
'''
df_train = df_train.na.fill(value=0)

# COMMAND ----------

'''
Checking If their are any repeated rows in the data set, we can see everything is at 1, so we dont have to drop any repeated rows
'''
df_train.groupBy('Page').count().display()

# COMMAND ----------

'''
Filter Different Types Of Languages:
zh - Chinese
en - English
ja - Japanese
de - German
es - Spanish
fr - French
ru - Russian
no_lang - Unidentified
'''

language_Chinese_zh = df_train.filter(df_train.Page.contains('zh')).display()
language_English_en = df_train.filter(df_train.Page.contains('en')).display()
language_Japanese_ja = df_train.filter(df_train.Page.contains('ja')).display()
language_German_de = df_train.filter(df_train.Page.contains('de')).display()
language_Spanish_es = df_train.filter(df_train.Page.contains('es')).display()
language_French_fr = df_train.filter(df_train.Page.contains('fr')).display()
language_Russian_ru = df_train.filter(df_train.Page.contains('ru')).display()
language_UnIdentified_no_lang = df_train.filter(df_train.Page.contains('no_lang')).display()

# COMMAND ----------

'''
Converting the Spark Dataframe into the Pandas dataframe for implemeting the forcasting model
'''

df = df_train.select("*").toPandas()
df

# COMMAND ----------

'''
Counting the Number of Wikipedia Pages per language to decide and train on the model based on more number of inputs
'''
def Count_LanguagePages(Page):
    value = re.search('[a-z][a-z].wikipedia.org',Page)
    if value:
        return value[0][0:2]           
    return 'no_lang'
df['language'] = df.Page.map(Count_LanguagePages)
print("Wikipedia Pages per Language are ", Counter(df.language))

# COMMAND ----------

'''
Filtering the datasets into the individual language Pandas Dataframes to train the Forcasting model based on the Language that would be needed

zh - Chinese
en - English
ja - Japanese
de - German
es - Spanish
fr - French
ru - Russian
no_lang - Unidentified
'''

ForModel = {}
ForModel['en'] = df[df.language=='en'].iloc[:,0:-1]
ForModel['fr'] = df[df.language == 'fr'].iloc[:, 0:-1]
ForModel['zh'] = df[df.language =='zh'].iloc[:,0:-1]
ForModel['de'] = df[df.language == 'de'].iloc[:, 0:-1]
ForModel['es'] = df[df.language == 'es'].iloc[:, 0:-1]
ForModel['ja'] = df[df.language == 'ja'].iloc[:, 0:-1]
ForModel['ru'] = df[df.language == 'ru'].iloc[:, 0:-1]
ForModel['no_lang'] = df[df.language == 'no_lang'].iloc[:, 0:-1]

'''
For Printing all the individual Dataframes above and Row Number is for the count of number of wikipedia pages in total
'''
for rowNumber in ForModel:
    print("Row Number:", ForModel[rowNumber],"\n")

# COMMAND ----------

'''
Sum of Views of each individual day in the dataset from 2015-07-02 to 2016-12-31
'''
PerDaySum = {} 
for rowNumber in ForModel:
    PerDaySum[rowNumber]=ForModel[rowNumber].iloc[:, 1:].sum(axis=0)/ForModel[rowNumber].shape[0]
for rowNumber in ForModel:
    print("rowNumber:", rowNumber)
    print("Per Day Views of all languages", PerDaySum[rowNumber])

# COMMAND ----------

'''
We Counted the Number of Pages that are highly viewed for each language avaliable in the dataset 

zh - Chinese
en - English
ja - Japanese
de - German
es - Spanish
fr - French
ru - Russian
no_lang - Unidentified
'''
HighestViewed = {}
for rowNumber in ForModel:
    print(rowNumber)
    PageCount = pd.DataFrame(ForModel[rowNumber][['Page']])
    PageCount['Total Views'] = ForModel[rowNumber].sum(axis=1)
    PageCount = PageCount.sort_values('Total Views',ascending=False)
    print(PageCount.head(5))
    HighestViewed[rowNumber] = PageCount.index[0]

# COMMAND ----------

'''
Plot For Number of Views per day of all languages
'''

plt.figure(figsize=(20, 12))
days = [r for r in range(PerDaySum['en'].shape[0])]
labels={'en':'English','zh':'Chinese','ja':'Japanese','no_lang':'Unidentified','fr':'French','de':'German','ru':'Russian','es':'Spanish'}
for rowNumber in PerDaySum:
  sns.lineplot(days, PerDaySum[rowNumber],label = labels[rowNumber])
  
plt.xlabel('Days (per Day)')
plt.ylabel('Per Wikipedia Page')
plt.title('All Wikipedia Pages (Off All Languages) Views Per Day')
plt.legend(loc = 'upper right')
plt.show()

# COMMAND ----------

'''
Arima Model Training based on the language 
'''


# COMMAND ----------

'''
Starting with English since it was the most views as we have seen in the ("All Wikipedia Pages (Off All Languages) Views Per Day") graph
'''

'''
Per Day Views Only for English
'''
plt.figure(figsize=(20, 12))
plt.xlabel('Days (per Day)')
plt.ylabel('Per English language Wikipedia Page')
plt.plot(days,PerDaySum['en'],label=labels['en'], color = 'green')
plt.show()

# COMMAND ----------

'''
Here We checked the rolling Mean and rolling standard deviation to check whether the graph is stationary or not

So we also implemented the Dickey Fuller test and Displayed the critical values
'''
def stats(x):
  '''
  Rolling Mean and Rolling Standard Deviation
  '''
  RollingMean = x.rolling(window=22,center=False).mean()
  RollingSTD = x.rolling(window=12,center=False).std()
  plt.figure(figsize=(20, 12))
  orig = plt.plot(x.values, color='yellow',label='Inputed')
  mean = plt.plot(RollingMean.values, color='green', label='RollingMean')
  std = plt.plot(RollingSTD.values, color='black', label = 'RollingSTD')
  plt.legend(loc = 'upper right')
  plt.title('Rolling Mean -- Rolling Standard Deviation')
  plt.show(block=False)
  '''
  Augmented Dickey-Fuller Test for finding the statistic and Derive P-Value
  '''
  ADF=adfuller(x)
  pvalue=ADF[2]
  print('Dickey Fuller Stastistic: %f'%ADF[0])
  print('p-value: %f'%ADF[1])
  for rowNumber,value in ADF[4].items():
     if ADF[0]>value:
        print("Not stationary Values")
        break
     else:
        print("Stationary Values")
        break;
  print("Critical Value:")
  for rowNumber,value in ADF[4].items():
    print('%s: %.2f' % (rowNumber, value))
stats(PerDaySum['en'])

# COMMAND ----------

"""
Since The P-value is greater than the 0.05 (Here P-Value We got is 0.19), So the Hypothesis is rejected, so we have to do log transformations to reduce the values and make the p value less than 0.05 (this helps to bring better predictions)
"""

# COMMAND ----------

'''
first try on Log transformations to bring the data to stationary
'''
log_transformations = np.log(PerDaySum['en'])
log_transformations_difference = log_transformations - log_transformations.shift()
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'black')
plt.show()

# COMMAND ----------

'''
Displaying the stats method that was used above to check if the values are converted into stationary or not after doing the log transformation
'''
log_transformations_difference.dropna(inplace=True)
stats(log_transformations_difference)

# COMMAND ----------

"""
Now we can see the P value is less than 0.05 we can say the values are stationary, we can now start implementing the ARIMA model to made the predictions
"""

# COMMAND ----------

"""
Did auto correlation and partial auto correlation to determine the values of Moving Average and Auto Regression for ARIMA model if need for better predictions, here we took 30 lags
"""

ACR = acf(log_transformations_difference, nlags=30)
PACR = pacf(log_transformations_difference, nlags=30)
plt.figure(figsize=(20, 12))
plt.title('Correlation Graphs')
plt.plot(ACR, color = 'red')
plt.plot(PACR, color = 'black')

# COMMAND ----------

'''
Auto Regressive Integrated Moving Average Model

initially we took moving average and auto regression values as one 
'''

ArimaModelImplementation = sm.tsa.ARIMA(log_transformations.values, order=(1,1,1))
ArimaModelResults = ArimaModelImplementation.fit(disp=-1) 
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'yellow')
plt.plot(ArimaModelResults.fittedvalues, color='red')
plt.title('Resudial Sum of Sqaure: %.2f'% sum((ArimaModelResults.fittedvalues-log_transformations_difference.values)**2))
plt.show()

# COMMAND ----------

'''
Summary of ARMIA
'''

print(ArimaModelResults.summary())

# COMMAND ----------

'''
Calculated the Error rate
'''

DataSplit = int(len(log_transformations)-100)
TrainSplit, TestSplit = log_transformations[0:DataSplit], log_transformations[DataSplit:len(log_transformations)]
GivenData_ = [x for x in TrainSplit]

GivenValues = list()
PredictionValues = list()
ErrorDF_en = list()

print('Predicted v Original')
print('\n')

for a in range(len(TestSplit)):
    ArimaModelImplementation = sm.tsa.ARIMA(GivenData_, order=(1, 1, 1))
    FittingModel = ArimaModelImplementation.fit(disp=0)
    print(FittingModel)
    output = FittingModel.forecast()
    
    Forecast = output[0]    
    True_Val = TestSplit[a]
    GivenData_.append(True_Val)
    Forecast = np.exp(Forecast)
    True_Val = np.exp(True_Val)
    
    Error_Val = ((abs(Forecast - True_Val))/True_Val)*100
    ErrorDF_en.append(Error_Val)
    print('predicted=%f,expected = %f,Error_Val=%f'%(Forecast,True_Val,Error_Val),'%')
    PredictionValues.append(float(Forecast))
    GivenValues.append(float(True_Val))

# COMMAND ----------

MeanOfError_en = sum(ErrorDF_en)/len(ErrorDF_en)
MeanOfError_en

# COMMAND ----------

plt.figure(figsize=(20, 12))
plt.plot(ErrorDF_en, label = "Error %", color = 'black')
plt.axhline(y= MeanOfError_en, linestyle='--',color='yellow')

# COMMAND ----------

"""
Predicted values to the original values on the testing dataset
"""
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]

plt.plot(test_day, GivenValues, color = 'green')
plt.plot(test_day, PredictionValues, color= 'black')
labels= {'Predicted','Inputed'}
plt.title('Inputed v Predicted for English')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()


# COMMAND ----------

'''
On overall dataset, we can see the model predicted better with only error rate around 5.4% for language 'en - ENGLISH'
'''
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]
labels={'Inputed','Predicted/Forecast'}
plt.plot(test_day, PredictionValues, color= 'green')
plt.plot(days, PerDaySum['en'], color = 'black')
plt.title('Inputed v Predicted/Forecast for English')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()

# COMMAND ----------

'''
We forcasted for english above and continued for some of the other languages
'''

# COMMAND ----------

'''
For Spanish (es)
Per Day Views Only for Spanish
'''

plt.figure(figsize=(20, 12))
plt.xlabel('Days (per Day)')
plt.ylabel('Per Spanish language Wikipedia Page')
plt.plot(days,PerDaySum['es'], color = 'green')
plt.show()

# COMMAND ----------

'''
Statistics
'''

stats(PerDaySum['es'])

# COMMAND ----------

'''
Log transformations to bring the data to stationary
'''
log_transformations = np.log(PerDaySum['es'])
log_transformations_difference = log_transformations - log_transformations.shift()
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'black')
plt.show()

# COMMAND ----------

'''
Displaying the stats method that was used above to check if the values are converted into stationary or not after doing the log transformation
'''
log_transformations_difference.dropna(inplace=True)
stats(log_transformations_difference)

# COMMAND ----------

"""
Did auto correlation and partial auto correlation to determine the values of Moving Average and Auto Regression for ARIMA model if need for better predictions, here we took 30 lags
"""

ACR = acf(log_transformations_difference, nlags=30)
PACR = pacf(log_transformations_difference, nlags=30)
plt.figure(figsize=(20, 12))
plt.title('Correlation Graphs')
plt.plot(ACR, color = 'red')
plt.plot(PACR, color = 'black')

# COMMAND ----------

'''
Auto Regressive Integrated Moving Average Model

initially we took moving average and auto regression values as one 
'''

ArimaModelImplementation = sm.tsa.ARIMA(log_transformations.values, order=(1,1,1))
ArimaModelResults = ArimaModelImplementation.fit(disp=-1) 
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'yellow')
plt.plot(ArimaModelResults.fittedvalues, color='red')
plt.title('Resudial Sum of Sqaure: %.2f'% sum((ArimaModelResults.fittedvalues-log_transformations_difference.values)**2))
plt.show()

# COMMAND ----------

print(ArimaModelResults.summary())

# COMMAND ----------

'''
Calculated the Error rate
'''

DataSplit = int(len(log_transformations)-100)
TrainSplit, TestSplit = log_transformations[0:DataSplit], log_transformations[DataSplit:len(log_transformations)]
GivenData_ = [x for x in TrainSplit]

GivenValues = list()
PredictionValues = list()
ErrorDF_es = list()

print('Predicted v Original')
print('\n')

for a in range(len(TestSplit)):
    ArimaModelImplementation = sm.tsa.ARIMA(GivenData_, order=(1, 1, 1))
    FittingModel = ArimaModelImplementation.fit(disp=0)
    print(FittingModel)
    output = FittingModel.forecast()
    
    Forecast = output[0]    
    True_Val = TestSplit[a]
    GivenData_.append(True_Val)
    Forecast = np.exp(Forecast)
    True_Val = np.exp(True_Val)
    
    Error_Val = ((abs(Forecast - True_Val))/True_Val)*100
    ErrorDF_es.append(Error_Val)
    print('predicted=%f,expected = %f,Error_Val=%f'%(Forecast,True_Val,Error_Val),'%')
    PredictionValues.append(float(Forecast))
    GivenValues.append(float(True_Val))

# COMMAND ----------

MeanOfError_es = sum(ErrorDF_es)/len(ErrorDF_es)
MeanOfError_es

# COMMAND ----------

plt.figure(figsize=(20, 12))
plt.plot(ErrorDF_es, label = "Error %", color = 'black')
plt.axhline(y= MeanOfError_es, linestyle='--',color='yellow')

# COMMAND ----------

"""
Predicted values to the original values on the testing dataset
"""
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]

plt.plot(test_day, GivenValues, color = 'green')
plt.plot(test_day, PredictionValues, color= 'black')
labels= {'Predicted','Inputed'}
plt.title('Inputed v Predicted for Spanish')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()


# COMMAND ----------

'''
On overall dataset, we can see the model predicted better with only error rate around 10.8% for language 'es-Spanish'
'''
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]
labels={'Inputed','Predicted/Forecast'}
plt.plot(test_day, PredictionValues, color= 'green')
plt.plot(days, PerDaySum['es'], color = 'black')
plt.title('Inputed v Predicted/Forecast for Spanish')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()

# COMMAND ----------

"""
Chinese
"""

# COMMAND ----------

'''
For chinese zh
Per Day Views Only for chinese
'''

plt.figure(figsize=(20, 12))
plt.xlabel('Days (per Day)')
plt.ylabel('Per Chinese language Wikipedia Page')
plt.plot(days,PerDaySum['zh'], color = 'green')
plt.show()

# COMMAND ----------

'''
Statistics
'''

stats(PerDaySum['zh'])

# COMMAND ----------

'''
Log transformations to bring the data to stationary
'''
log_transformations = np.log(PerDaySum['zh'])
log_transformations_difference = log_transformations - log_transformations.shift()
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'black')
plt.show()

# COMMAND ----------

'''
Displaying the stats method that was used above to check if the values are converted into stationary or not after doing the log transformation
'''
log_transformations_difference.dropna(inplace=True)
stats(log_transformations_difference)

# COMMAND ----------

"""
Did auto correlation and partial auto correlation to determine the values of Moving Average and Auto Regression for ARIMA model if need for better predictions, here we took 30 lags
"""

ACR = acf(log_transformations_difference, nlags=30)
PACR = pacf(log_transformations_difference, nlags=30)
plt.figure(figsize=(20, 12))
plt.title('Correlation Graphs')
plt.plot(ACR, color = 'red')
plt.plot(PACR, color = 'black')

# COMMAND ----------

'''
Auto Regressive Integrated Moving Average Model

initially we took moving average and auto regression values as one 
'''

ArimaModelImplementation = sm.tsa.ARIMA(log_transformations.values, order=(1,1,1))
ArimaModelResults = ArimaModelImplementation.fit(disp=-1) 
plt.figure(figsize=(20, 12))
plt.plot(log_transformations_difference.values, color = 'yellow')
plt.plot(ArimaModelResults.fittedvalues, color='red')
plt.title('Resudial Sum of Sqaure: %.2f'% sum((ArimaModelResults.fittedvalues-log_transformations_difference.values)**2))
plt.show()

# COMMAND ----------

print(ArimaModelResults.summary())

# COMMAND ----------

'''
Calculated the Error rate
'''

DataSplit = int(len(log_transformations)-100)
TrainSplit, TestSplit = log_transformations[0:DataSplit], log_transformations[DataSplit:len(log_transformations)]
GivenData_ = [x for x in TrainSplit]

GivenValues = list()
PredictionValues = list()
ErrorDF_zh = list()

print('Predicted v Original')
print('\n')

for a in range(len(TestSplit)):
    ArimaModelImplementation = sm.tsa.ARIMA(GivenData_, order=(1, 1, 1))
    FittingModel = ArimaModelImplementation.fit(disp=0)
    print(FittingModel)
    output = FittingModel.forecast()
    
    Forecast = output[0]    
    True_Val = TestSplit[a]
    GivenData_.append(True_Val)
    Forecast = np.exp(Forecast)
    True_Val = np.exp(True_Val)
    
    Error_Val = ((abs(Forecast - True_Val))/True_Val)*100
    ErrorDF_zh.append(Error_Val)
    print('predicted=%f,expected = %f,Error_Val=%f'%(Forecast,True_Val,Error_Val),'%')
    PredictionValues.append(float(Forecast))
    GivenValues.append(float(True_Val))

# COMMAND ----------

MeanOfError_zh = sum(ErrorDF_zh)/len(ErrorDF_zh)
MeanOfError_zh

# COMMAND ----------

plt.figure(figsize=(20, 12))
plt.plot(ErrorDF_zh, label = "Error %", color = 'black')
plt.axhline(y= 6.21642082, linestyle='--',color='yellow')

# COMMAND ----------

"""
Predicted values to the original values on the testing dataset
"""
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]

plt.plot(test_day, GivenValues, color = 'green')
plt.plot(test_day, PredictionValues, color= 'black')
labels= {'Predicted','Inputed'}
plt.title('Inputed v Predicted for Chinese')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()


# COMMAND ----------

'''
On overall dataset, we can see the model predicted better with only error rate around 10.8% for language 'zh-Chinese'
'''
plt.figure(figsize=(20, 12))
test_day = [a+500 for a in range(len(TestSplit))]
labels={'Inputed','Predicted/Forecast'}
plt.plot(test_day, PredictionValues, color= 'green')
plt.plot(days, PerDaySum['zh'], color = 'black')
plt.title('Inputed v Predicted/Forecast for Chinese')
plt.xlabel('Per Day')
plt.ylabel('Views')
plt.legend(labels)
plt.show()

# COMMAND ----------

plt.axhline(y= MeanOfError_zh, linestyle='--',color='green')
plt.axhline(y= MeanOfError_en, linestyle='--',color='black')
plt.axhline(y= MeanOfError_es, linestyle='--',color='blue')
plt.title('Comparing Error Rates of different languages')

# COMMAND ----------

print(f'Accuracy of English{100-MeanOfError_en}')
print(f'Accuracy of Spanish{100-MeanOfError_es}')
print(f'Accuracy of Chinese{100-MeanOfError_zh}')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['English', 'Spanish', 'Chinese']
plt.title('Comparing Accuracy of different languages')
Error_Rate = [94.59,89.197,93.78]
ax.bar(langs,Error_Rate)
plt.show()

# COMMAND ----------


