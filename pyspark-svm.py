from __future__ import print_function
# import pyspark
from pyspark     import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.classification import SVMModel
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
#import numpy as np

#import os !
import sys
import os

if __name__ == "__main__":
    
    #Initialize--初始化 
    local_input_file = str(sys.argv[1]) if len(sys.argv) > 1 else "hdfs:///input/stock_data.csv"
    #conf = SparkConf().setMaster("local").setAppName("My App")###local test 本地测试
    #sc = SparkContext(conf = conf)###local test 本地测试
    sc = SparkContext(appName='SVM for stock SSE')###yarn 集群测试
    spark = SQLContext(sc)
    
    #Read data from hdfs and create Spark DataFrame--读取hdfs的csv文件并创建Spark DataFrame
    print("Start loading data... \n开始导入数据...\n")
    df_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('hdfs:///input/stock_train_data.csv')
    df_test = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('hdfs:///input/stock_test_data.csv')
    
    #Show a part of DataFrame--展示DataFrame
    print("To simplify the code, we have processed the data by MySQL")
    print("为了简化代码，我们已经使用MySQL进行了数据处理\n")
    print("This is part of train dataset of DataFrame \n展现部分训练集数据")
    df_train.show(5)
    print("This is part of test dataset of DataFrame \n展现部分测试集数据")
    df_test.show(5)
    print("This is the schema of DataFrame \n展现数据表结构")
    df_test.printSchema()
    
    #df.drop('age').collect()
    df_train = df_train.drop('trade_date')
    df_train = df_train.drop('ts_code')
    df_test = df_test.drop('ts_code')
    df_test = df_test.drop('trade_date')
    
    print("This is the data type of DataFrame \n展现数据格式")
    print(df_test.dtypes)

    """
    #Preprocesse data--数据预处理--(needs optimization without numpy-需优化至不用numpy)
    #np_data = np.array(df.values.collect())
    y = np.array(df.select("open").collect())
    x = np.array([])
    df.schema.names #get all key获取全部键值(列名)
    for i in df.schema.names:
        np.append(x,df.select(i).collect())
    
    #Match price for stock--匹配价格
    x = np.delete(x, 0, 0) # 1 to n close price 收盘价
    y = np.delete(y, len(x), 0)# 2 to open price 开盘价
    y = y.ravel()
    
    #Split data in time series--数据集分割按时间序列
    x_train,x_test = np.array_split(x, (170,))
    y_train,y_test = np.array_split(y, (170,))
    """
    #Preprocesse data--数据预处理

    
    #Convert DataFrame to LabeledPoint data--
    #将DataFrame转化为LabeledPoint类型的数据label和features
    train_data = sc.parallelize(df_train.collect()).map(lambda x:LabeledPoint(label=x[0],features=x[1:]))
    print("\nThis is the first RDD of train data \n展现首个训练集RDD数据")
    print(train_data.first())
    print("Number for train --训练集数量:{}".format(train_data.count()))
    
    test_data = sc.parallelize(df_test.collect()).map(lambda x:LabeledPoint(label=x[0],features=x[1:]))
    print("\nThis is the first RDD of test data \n展现首个测试集RDD数据")
    print(test_data.first())
    print("Number for test --测试集数量:{}".format(test_data.count()))

    #Train SVM model--训练模型
    svm = SVMWithSGD.train(train_data, iterations=500)
    #prediction = svm.predict(train_data.first().features)
    #print("\n真实值:{},预测值{}".format(train_data.first().label,prediction))

    #Find prediction accuracy--总体预测准确率
    svmTotalCorrect = test_data.map(lambda x: 1 if svm.predict(x.features) == x.label else 0).sum()
    print("\nClassification accuracy --分类准确数:{}".format(svmTotalCorrect))
    svmAccuracy = float(svmTotalCorrect)/test_data.count()
    print("\nForecast accuracy rate --总体预测准确率为:{}".format(svmAccuracy))
    
    #Calculate AUC--计算
    scoreAndLabels = test_data.map(lambda x:(float(svm.predict(x.features)),x.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    print('\nPR:{:.4f}, AUC:{:.4f}'.format(metrics.areaUnderPR, metrics.areaUnderROC))