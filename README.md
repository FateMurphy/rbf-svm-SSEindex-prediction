RBF-SVM SSE Prediction
========
Use RBF-SVM model to predict Shanghai Stock Exchange (SSE) Composite Index at Python and PySpark. 

#1. For Python
------
please enter your TUSHARE password or you can get a free account at https://waditu.com/document/1    
Of course, you can also try using only part of the data provided by this project.   
```python
import tushare as ts #Python
ts.set_token(#################################)# enter your ts password here
```

For cross-validation methods, this project uses some special methods, you can refer to this website   
http://www.zhengwenjie.net/tscv/    
There are many initial values that need to be set at SVR part:    
```python
#initial values settings
G = [0.001,0.01,0.1,1,10,100,1000] #Initial grid search set for Gamma
C = G  #Initial grid search set for C
K = 40 #Cross-validation fold
gap = 4 #gap for cross-validation only for GKF
n = 100 #Multiple grid search loops
kind = 'KF' #Types of cross-validation: KF LKF OKF GKF
mod = 'MAE' #Cross-validation model evaluation method: RMSE MAE r2 MSE
period = 60 #Prediction period length (days)
```

#2. For Spark and Hadoop 
--------
**Warning!!!**: The data in the pyspark file in this project has been preprocessed, and the processing process can refer to the operation in above.   

PySpark is a Python tool to help you use Spark without Scala.   
In this project, only the data is classified as SVC in spark.   
Of course, you can also do an SVR with a linear kernel function with PySpark, but the effect is not estimated.   
PySpark does not seem to support the RBF kernel function yet.     
This part of the work in project is only implemented in part 1 of Python.    

PS: It is strongly recommended to install jupyter for your server/virtual machine before compiling PySpark
