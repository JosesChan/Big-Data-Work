from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer, OneHotEncoder
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
pandasNuclearPlantsSmall = pandas.read_csv("dataset/nuclear_plants_small_dataset.csv")
sparksNuclearPlantsSmall= spark.read.csv("dataset/nuclear_plants_small_dataset.csv", header=True,inferSchema=True)

# Task 1: Check to see if there are any missing values
# null values in each column

dataframeNaN = pandasNuclearPlantsSmall[pandasNuclearPlantsSmall.isna().any(axis=1)]
if(dataframeNaN.empty):
    print("No null values")
else:
    print("Null values\n")
    print(dataframeNaN)

# Task 2: Normal Group and Abnormal Group, find min, max, mean, median, mode, variance and boxplot
columnNames = pandasNuclearPlantsSmall.drop(["Status"],axis = 1).columns.values

pandasNormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Normal"]
pandasAbnormalNuclearPlantsSmall = pandasNuclearPlantsSmall.loc[pandasNuclearPlantsSmall["Status"] == "Abnormal"]


# for i in columnNames:
#     plt.figure()
#     seaborn.boxplot(x=pandasNuclearPlantsSmall["Status"], y=pandasNuclearPlantsSmall[i])
#     print("Normal" + i)
#     print("MINIMUM: \") 
#     print(pandasNormalNuclearPlantsSmall[i].max())
#     print("MAX: \") 
#     print(pandasNormalNuclearPlantsSmall[i].min())
#     print("MEAN: \") 
#     print(pandasNormalNuclearPlantsSmall[i].mean())
#     print("MEDIAN: \") 
#     print(pandasNormalNuclearPlantsSmall[i].median())
#     print("MODE: \") 
#     print(pandasNormalNuclearPlantsSmall[i].mode())
#     print("VAR: \") 
#     print(pandasNormalNuclearPlantsSmall[i].var())
#     print("Abnormal "+ i)
#     print("MINIMUM: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].max())
#     print("MAX: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].min())
#     print("MEAN: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].mean())
#     print("MEDIAN: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].median())
#     print("MODE: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].mode())
#     print("VAR: \") 
#     print(pandasAbnormalNuclearPlantsSmall[i].var())
#     print()
#     plt.savefig('Graph '+i)
# plt.show()

# Task 3: Show in a table the correlation matrix, where each element shows correlation between two features, find highly correlated features.

# Using pearson and not spearman's or Kendall's since data is not monotonic
# print("Pearson Correlation Matrix")
# print(pandasNormalNuclearPlantsSmall.corr(method="pearson"))
# print(pandasAbnormalNuclearPlantsSmall.corr(method="pearson"))

# Task 4: Shuffle data into 70% training set and 30% test set

# References
#https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
#https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa

# Index labels, adding metadata (for identification) to the label column
# Fit the entire dataset, including all labels in index
labelIndexer = StringIndexer(inputCol="Status", outputCol="statusIndexedLabel").fit(sparksNuclearPlantsSmall)

# Store list of original column names and data held by the columns
colNameList = sparksNuclearPlantsSmall.schema.names
cols = sparksNuclearPlantsSmall.columns

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

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=float("inf"))
stages += [normalizer]

# Pipeline to create stages
# Ensuring data goes through identical processing steps
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(sparksNuclearPlantsSmall)
df = pipelineModel.transform(sparksNuclearPlantsSmall)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Total Classified Samples), calculate sensitivity and specificity 

# Instance of Decision tree classifier
treeClf = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
# Train model, pipeline estimator stage producing model by fitting data onto class
decisionTreeModel = treeClf.fit(trainingData)
# Predict results, pipeline transformer stage where the model makes the predictions from the dataset
predictions = decisionTreeModel.transform(testData)
# Select example rows to display.
predictions.select(colNameList[0],colNameList[1],colNameList[2],'rawPrediction', 'prediction', 'probability').show(10)
# Compute error rate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Decision Tree Test Error = %g" % (1.0 - accuracy))


# Build the model
# svmModel = SVMWithSGD.train(trainingData, iterations=100)

# # Evaluating the model on training data
# labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
# print("Training Error = " + str(trainErr))

# # Save and load model
# svmModel.save(sc, "target/tmp/pythonSVMWithSGDModel")
# sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")


# Instance of Support Vector Machines classifier
svmClf = LinearSVC(maxIter=10, regParam=0.1)
# Train model, pipeline estimator stage producing model by fitting data onto class
lsvcModel = svmClf.fit(trainingData)
# Predict results, pipeline transformer stage where the model makes the predictions from the dataset
predictions = lsvcModel.transform(testData)
# Compute error rate
# Show the computed predictions and compare with the original labels
predictions.select("features", "label", "prediction").show(10)
# Define the evaluator method with the corresponding metric and compute the classification error on test data
evaluator = MulticlassClassificationEvaluator().setMetricName('accuracy')
accuracy = evaluator.evaluate(predictions) 
# Show the accuracy
print ("Support Vector Machine Test Error = %g" % (1.0 - accuracy))


# Instance of multilayer perceptron, type of artificial neural network
# annClf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# annClf.fit(xTrain, yTrain)
# yPredictAnnClf = annClf.predict(xTest)
# accuracy = zero_one_loss(yTest, yPredictAnnClf)
# annErrorRate = 1 - accuracy


# # Task 6: Compare results based on task 5, which is best

# print("Error rate for decision tree" + treeErrorRate)
# print("Error rate for support vector machines" + svmErrorRate)
# print("Error rate for artificial neural network" + annErrorRate)

# Task 7: Discuss if features can detect abnormality in reactors
# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature

