from pyspark.sql.functions import *
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
import pyspark.SparkContext
import pandas 
import seaborn
import matplotlib.pyplot as plt
import pyspark.sql.functions


spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
pandasNuclearPlantsSmall = pandas.read_csv("dataset/nuclear_plants_small_dataset.csv")
sparksNuclearPlantsSmall= spark.read.csv("dataset/nuclear_plants_small_dataset.csv", header=True,inferSchema=True)

# Task 1: Check to see if there are any missing values
# null values in each column

# dataframeNaN = pandasNuclearPlantsSmall[pandasNuclearPlantsSmall.isna().any(axis=1)]
# if(dataframeNaN.empty):
#     print("No null values")
# else:
#     print("Null values\n")
#     print(dataframeNaN)

# Task 2: Normal Group and Abnormal Group, find min, max, mean, median, mode, variance and boxplot
featureNames = pandasNuclearPlantsSmall.drop(["Status"],axis = 1).columns.values

pandasNormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Normal"]
pandasAbnormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Abnormal"]


# for i in featureNames:
#     plt.figure()
#     seaborn.boxplot(x=pandasNuclearPlantsSmall["Status"], y=pandasNuclearPlantsSmall[i])
#     print("Normal" + i)
#     print("MINIMUM: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].max())
#     print("MAX: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].min())
#     print("MEAN: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].mean())
#     print("MEDIAN: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].median())
#     print("MODE: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].mode())
#     print("VAR: \n") 
#     print(pandasNormalNuclearPlantsSmall[i].var())
#     print("Abnormal "+ i)
#     print("MINIMUM: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].max())
#     print("MAX: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].min())
#     print("MEAN: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].mean())
#     print("MEDIAN: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].median())
#     print("MODE: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].mode())
#     print("VAR: \n") 
#     print(pandasAbnormalNuclearPlantsSmall[i].var())
#     print()
#     plt.savefig('Graph '+i)
# plt.show()

# Data shows a large amount of outliers that can affect the calculations, robustscaler should be used


# Task 3: Show in a table the correlation matrix, where each element shows correlation between two features, find highly correlated features.

# Using pearson and not spearman's or Kendall's since data is not monotonic
# print("Pearson Correlation Matrix")
# print(pandasNormalNuclearPlantsSmall.corr(method="pearson"))
# print(pandasAbnormalNuclearPlantsSmall.corr(method="pearson"))

# Task 4: Shuffle data into 70% training set and 30% test set

# References
#https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
#https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa

# Store list of original column names and data held by the columns
colNameList = sparksNuclearPlantsSmall.schema.names
cols = sparksNuclearPlantsSmall.columns
seedNumber = 1234

# Link the stages into a pipeline
stages = []

# Feature extractor
# String Indexer
for columns in colNameList:
    stringIndexer = StringIndexer(inputCol = columns, outputCol = columns + 'Index')
    stages += [stringIndexer]

label_stringIdx = StringIndexer(inputCol = 'Status', outputCol = 'label')
stages += [label_stringIdx]
numericColsNameList = (sparksNuclearPlantsSmall.drop("Status")).schema.names

# Feature transformer
# Vector Assembler
assembler = VectorAssembler(inputCols=numericColsNameList, outputCol="features")
stages += [assembler]

# Robust Scaler, an estimator that 
scaler = RobustScaler(inputCol="features", outputCol="scaledFeatures", withScaling=True, withCentering=False, lower=0.25, upper=0.75)
stages += [scaler]

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=(2))
stages += [normalizer]

# Normalize each Vector using $L^1$ norm.
normalizerScaled = Normalizer(inputCol="scaledFeatures", outputCol="normRobScaleFeatures", p=(2))
stages += [normalizerScaled]


# Pipeline to create stages
# Ensuring data goes through identical processing steps
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(sparksNuclearPlantsSmall)
df = pipelineModel.transform(sparksNuclearPlantsSmall)
selectedCols = ['label', 'features', 'normFeatures', 'scaledFeatures', 'normRobScaleFeatures'] + cols
df = df.select(selectedCols)
# df.printSchema()

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=seedNumber)

# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Total Classified Samples), calculate sensitivity and specificity 

# Instance of Decision tree classifier
treeClf = DecisionTreeClassifier(featuresCol = 'scaledFeatures', labelCol = 'label')
# Train model, pipeline estimator stage producing model by fitting data onto class
decisionTreeModel = treeClf.fit(trainingData)
# Predict results, pipeline transformer stage where the model makes the predictions from the dataset
predictions = decisionTreeModel.transform(testData)
# Compute error rate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Decision Tree Test Error = %g" % (1.0 - accuracy))

# Instance of Support Vector Machines classifier
svmClf = LinearSVC(featuresCol = 'scaledFeatures', labelCol = 'label')
# Train model, pipeline estimator stage producing model by fitting data onto class
lsvcModel = svmClf.fit(trainingData)
# Predict results, pipeline transformer stage where the model makes the predictions from the dataset
predictions = lsvcModel.transform(testData)
# Compute error rate
evaluator = MulticlassClassificationEvaluator().setMetricName('accuracy')
accuracy = evaluator.evaluate(predictions) 
# Show the accuracy
print ("Support Vector Machine Test Error = %g" % (1.0 - accuracy))


# Instance of multilayer perceptron, type of artificial neural network
# input layer of size 12 (features), 2 hidden layers of size 8 and output of size 2 (classes)
layers = [12, 8, 8, 2]

annClf = MultilayerPerceptronClassifier(featuresCol = 'scaledFeatures', labelCol = 'label', layers=layers, seed=seedNumber)
annModel = annClf.fit(trainingData)
predictions = annModel.transform(testData)
accuracy = evaluator.evaluate(predictions) 
annErrorRate = 1 - accuracy
# Show the accuracy
print ("Multilayer perceptron Test Error = %g" % (1.0 - accuracy))

# # Task 6: Compare results based on task 5, which is best

# Task 7: Discuss if features can detect abnormality in reactors

# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature

# mapper: splits a single tweet string into words, cleans them, and returns a Counter


# MapReduce
def reduceFunc(accum, n):
    print(accum, n)
    if accum[1] > n[1]:
        return(n)
    else: return(accum)

def mapFunc(lines):
    return (lines[0], lines[1])

rdd = sc.parallelize([("Mumbai", 19, 30),
    ("Delhi", 5, 41),
    ("Kolkata", 20, 40),
    ("Mumbai", 18, 35),
    ("Delhi", 4, 42),
    ("Delhi", 10, 44),
    ("Kolkata", 19, 39)])

rdd.map(mapFunc).keyBy(lambda x: x[0]).reduceByKey(reduceFunc).map(lambda x : x[1]).collect()


dfMap = sparksNuclearPlantsSmall.map(lambda x: (x[0], tuple(x[1:])))

cityTempMin = dfMap.reduceByKey(lambda x, y: min(x[0],y[0]))

cityTempMin.collect()


#  map (in_key, in_value) -> list (out_key, intermediate_value

# ● reduce (out_key, list(intermediate_value)) -> list (out_value)

# reducer: merges two Counter objects given as arguments (hint: use update)
# chunk_mapper: applies mapper and reducer to a given chunk of tweets, and returns the result
