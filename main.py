# References
#https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
#https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
#https://uk.mathworks.com/help/matlab/import_export/compute-mean-value-with-mapreduce.html
#https://spark.apache.org/docs/1.1.1/api/python/pyspark.rdd.RDD-class.html
#https://hackersandslackers.com/working-with-pyspark-rdds

from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark import RDD, SparkContext, SparkConf
from pyspark.sql.context import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer, OneHotEncoder, RobustScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import pandas 
import seaborn
import matplotlib.pyplot as plt
import pyspark.sql.functions


spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
pandasNuclearPlantsSmall = pandas.read_csv("nuclear_plants_small_dataset.csv")
sparksNuclearPlantsSmall= spark.read.csv("nuclear_plants_small_dataset.csv", header=True,inferSchema=True)
# sparksNuclearPlantsLarge = spark.read.csv("dataset/nuclear_plants_large_dataset.csv", header=True,inferSchema=True)


# Task 1: Check to see if there are any missing values
# null values in each column

dataframeNaN = pandasNuclearPlantsSmall[pandasNuclearPlantsSmall.isna().any(axis=1)]
if(dataframeNaN.empty):
    print("No null values")
else:
    print("Null values\n")
    pandasNuclearPlantsSmall = pandasNuclearPlantsSmall[pandasNuclearPlantsSmall.notna().any(axis=1)]
    print(dataframeNaN)

# Task 2: Normal Group and Abnormal Group, find min, max, mean, median, mode, variance and boxplot
featureNames = pandasNuclearPlantsSmall.drop(["Status"],axis = 1).columns.values

pandasNormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Normal"]
pandasAbnormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Abnormal"]


for i in featureNames:
    plt.figure()
    seaborn.boxplot(x=pandasNuclearPlantsSmall["Status"], y=pandasNuclearPlantsSmall[i])
    print("Normal" + i)
    print("MINIMUM: \n") 
    print(pandasNormalNuclearPlantsSmall[i].max())
    print("MAX: \n") 
    print(pandasNormalNuclearPlantsSmall[i].min())
    print("MEAN: \n") 
    print(pandasNormalNuclearPlantsSmall[i].mean())
    print("MEDIAN: \n") 
    print(pandasNormalNuclearPlantsSmall[i].median())
    print("MODE: \n") 
    print(pandasNormalNuclearPlantsSmall[i].mode())
    print("VAR: \n") 
    print(pandasNormalNuclearPlantsSmall[i].var())
    print("Abnormal "+ i)
    print("MINIMUM: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].max())
    print("MAX: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].min())
    print("MEAN: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].mean())
    print("MEDIAN: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].median())
    print("MODE: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].mode())
    print("VAR: \n") 
    print(pandasAbnormalNuclearPlantsSmall[i].var())
    print()
    plt.savefig('Graph '+i)
plt.show()

# Task 3: Show in a table the correlation matrix, where each element shows correlation between two features, find highly correlated features.

# # Using pearson and not spearman's or Kendall's since data is not monotonic
print("Pearson Correlation Matrix")
print(pandasNormalNuclearPlantsSmall.corr(method="pearson"))
print(pandasAbnormalNuclearPlantsSmall.corr(method="pearson"))

# Task 4: Shuffle data into 70% training set and 30% test set
seedNumber = 1234
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = sparksNuclearPlantsSmall.randomSplit([0.7, 0.3], seed=seedNumber)

# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Total Classified Samples), calculate sensitivity and specificity 
# Store list of original column names and data held by the columns
colNameList = sparksNuclearPlantsSmall.schema.names
cols = sparksNuclearPlantsSmall.columns

# Link the stages into a ETL pipeline
stages = []

# Feature extractor
# String Indexer
label_stringIdx = StringIndexer(inputCol = 'Status', outputCol = 'label')
stages += [label_stringIdx]
numericColsNameList = (sparksNuclearPlantsSmall.drop("Status")).schema.names

# Feature transformer
# Vector Assembler
assembler = VectorAssembler(inputCols=numericColsNameList, outputCol="features")
stages += [assembler]

# Data shows a large amount of outliers that can affect the calculations, robustscaler should be used
scaler = RobustScaler(inputCol="features", outputCol="scaledFeatures", withScaling=True, withCentering=False, lower=0.25, upper=0.75)
stages += [scaler]

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=(2))
stages += [normalizer]

# Normalize each Vector using $L^1$ norm.
normalizerScaled = Normalizer(inputCol="scaledFeatures", outputCol="normRobScaleFeatures", p=(2))
stages += [normalizerScaled]


def classifierChoice(classifierChosen):
    if(classifierChosen==1):
        print("Decision Tree Results:")
        treeModel=DecisionTreeClassifier(featuresCol = 'scaledFeatures', labelCol = 'label')
        return([treeModel])
    if(classifierChosen==2):
        print("Linear Support Vector Machine Results:")
        lsvcModel=LinearSVC(featuresCol = 'scaledFeatures', labelCol = 'label')
        return([lsvcModel])
    if(classifierChosen==3):
        print("MultilayerPerceptron Results:")
        annModel=MultilayerPerceptronClassifier(featuresCol = 'scaledFeatures', labelCol = 'label', layers=[12, 8, 8, 2], seed=seedNumber)
        return([annModel])

# Pipeline to create stages
# Ensuring data goes through identical processing steps
def pipelineActivate(stages, classifierChoice):
    pipeline = Pipeline(stages = stages+classifierChoice)
    pipelineModel = pipeline.fit(trainingData)
    predictions = pipelineModel.transform(trainingData)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print ("Test Error = %g" % (1.0 - accuracy))

pipelineActivate(stages, classifierChoice(1))
pipelineActivate(stages, classifierChoice(2))
pipelineActivate(stages, classifierChoice(3))

# Task 6: Compare results based on task 5, which is best

# Task 7: Discuss if features can detect abnormality in reactors

# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature
#RUN IN GOOGLE COLLAB

nuclearLarge = spark.read.csv("nuclear_plants_big_dataset.csv", header=True,inferSchema=True)
nuclearLarge = nuclearLarge.drop("Status")
colNamesLarge = nuclearLarge.schema.names

nuclearLargeRdd = nuclearLarge.rdd

nuclearLargeRddCurrent = nuclearLargeRdd.map(lambda x: x.Power_range_sensor_1)

# find minimum, if x is less than y return x else return y, aggregate elements using this function
minimum = nuclearLargeRddCurrent.reduce(lambda x, y: x if (x < y) else y)
# find maximum, if x is more than y return x else return y, aggregate elements using this function
maximum = nuclearLargeRddCurrent.reduce(lambda x, y: x if (x > y) else y)

# find mean
meanVal = nuclearLargeRddCurrent.reduce(lambda x, y: x+y)
meanVal = meanVal/nuclearLargeRddCurrent.count()
    
print("Current Sensor: ")
print("Minimum: ")
print(minimum)
print("Maximum: ")
print(maximum)
print("Mean: ")
print(meanVal)