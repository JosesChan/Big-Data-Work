
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pandas 
import seaborn
import matplotlib.pyplot as plt

# library & dataset
import seaborn as sns
#df = sns.load_dataset('iris')

#sns.boxplot( x=df["species"], y=df["sepal_length"] )
#plt.show()

#pyspark matlablib and seaborn pandas

spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
sparksNuclearPlantsSmall = spark.read.csv("dataset/nuclear_plants_small_dataset.csv")

#ratings = ratings.withColumn('userId', col('userId').cast('integer'))

pandasNuclearPlantsSmall = pandas.read_csv("dataset/nuclear_plants_small_dataset.csv")
#sparksNuclearPlantsSmall.toPandas()



# Task 1: Check to see if there are any missing values

#emptyDataRows = nuclearPlantsSmall[nuclearPlantsSmall.isnull().any(axis=1)]

# Task 2: Normal Group and Abnormal Group, find min, max, mean, median, mode, variance and boxplot
pandasNormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Normal"]
pandasAbnormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Abnormal"]

seaborn.boxplot(x="Normal Sensors",data=pandasNormalNuclearPlantsSmall)
plt.show()

# Task 3: Show in a table the correlation matrix, where each element shows correlation between two features, find highly correlated features.
# Task 4: Shuffle data into 70% training set and 30% test set
# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Tota; Classified Samples), calculate sensitivity and specificity 
# Task 6: Compare results based on task 5, which is best
# Task 7: Discuss if features can detect abnormality in reactors
# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature

