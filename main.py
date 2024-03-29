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
import numpy
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
    print("\nNormal" + i)
    print("MINIMUM: ") 
    print(pandasNormalNuclearPlantsSmall[i].min())
    print("MAX: ") 
    print(pandasNormalNuclearPlantsSmall[i].max())
    print("MEAN: ") 
    print(pandasNormalNuclearPlantsSmall[i].mean())
    print("MEDIAN: ") 
    print(pandasNormalNuclearPlantsSmall[i].median())
    print("MODE: ") 
    print(pandasNormalNuclearPlantsSmall[i].mode())
    print("VAR: ") 
    print(pandasNormalNuclearPlantsSmall[i].var())
    print("\nAbnormal "+ i)
    print("MINIMUM: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].min())
    print("MAX: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].max())
    print("MEAN: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].mean())
    print("MEDIAN: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].median())
    print("MODE: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].mode())
    print("VAR: ") 
    print(pandasAbnormalNuclearPlantsSmall[i].var())
    print()
    plt.savefig('Graph '+i)
plt.show()

# Task 3: Show in a table the correlation matrix, where each element shows correlation between two features, find highly correlated features.

# # Using pearson and not spearman's or Kendall's since data is not monotonic
print("Pearson Correlation Matrix")
print(pandasNuclearPlantsSmall.corr(method="pearson"))

# Task 4: Shuffle data into 70% training set and 30% test set
seedNumber = 1234
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = sparksNuclearPlantsSmall.randomSplit([0.7, 0.3], seed=seedNumber)

# find where status equals group
normTrain = (trainingData.where(col("Status") == "Normal"))
abnormTrain = (trainingData.where(col("Status") == "Abnormal"))
normTest = (testData.where(col("Status")== "Normal"))
abnormTest = (testData.where(col("Status") == "Abnormal"))

#output dataset statistics
print("Training Data Records:")
print(trainingData.count())
print("Normal Status Records: ")
print(normTrain.count())
print("Abnormal Status Records: ")
print(abnormTrain.count())
print("Test Data Records: ")
print(testData.count())
print("Normal Status Records: ")
print(normTest.count())
print("Abnormal Status Records: ")
print(abnormTest.count())


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

# # Normalize features.
# normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=(2))
# stages += [normalizer]

# # Normalize scaled features
# normalizerScaled = Normalizer(inputCol="scaledFeatures", outputCol="normRobScaleFeatures", p=(2))
# stages += [normalizerScaled]


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
    
    truePositives = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 1)].count()
    trueNegatives = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 0)].count()
    falsePositives = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 1)].count()
    falseNegative = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 0)].count()


    print("Test Error = %g" % (1.0 - accuracy))
    print("Sensitivity = %g" % (truePositives/(truePositives+falseNegative)))
    print("Specificity = %g" % (trueNegatives/(trueNegatives+falsePositives)))

pipelineActivate(stages, classifierChoice(1))
pipelineActivate(stages, classifierChoice(2))
pipelineActivate(stages, classifierChoice(3))

# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature
# RUNNING IN GOOGLE COLLAB AS VSCODE IMPLEMENTATION WAS INOPERABLE

# Set up rdd and get col names and length of rdd
nuclearLarge = spark.read.csv("nuclear_plants_big_dataset.csv", header=True,inferSchema=True)
nuclearLarge = nuclearLarge.drop("Status")
colNamesLarge = nuclearLarge.schema.names
nuclearLargeRdd = nuclearLarge.rdd
iteratorColNames = 0

# function to input into max
def maxMapping(x):
    yield max(x)

def minMapping(x):
    yield min(x)

# sums chunks to arrays
def sumMapping(x): 
    sumArray = numpy.array((list(x)))
    yield numpy.sum(sumArray, 0)

# Map partitions operates across entire rdd using mapping function which will
# yield x 
maximumMap = nuclearLargeRdd.mapPartitions(maxMapping)
minimumMap = nuclearLargeRdd.mapPartitions(minMapping)
sumMap = nuclearLargeRdd.mapPartitions(sumMapping)

# Reduce between chunks for each partition
# Use function to determine minimum
maxReducer = maximumMap.reduce(lambda x, y: x if (x > y) else y)
# Use function to determine maximum
minReducer = minimumMap.reduce(lambda x, y: x if (x < y) else y)

# Sum arrays, axis equals 0 to show all columns/features
sums = sumMap.reduce(lambda x,y: numpy.sum([x,y], axis = 0))

# Count number of rows in the rdd
rddLength = nuclearLargeRdd.count()

# Output values
print("Maximum:")
print(maxReducer)
print("Minimum:")
print(minReducer)
print("Mean")
for i in sums:
    print(colNamesLarge[iteratorColNames])
    print(i/rddLength)
    iteratorColNames += 1