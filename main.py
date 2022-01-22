from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import pandas 
import seaborn
import matplotlib.pyplot as plt
import pyspark.sql.functions
spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
pandasNuclearPlantsSmall = pandas.read_csv("dataset/nuclear_plants_small_dataset.csv")
sparksNuclearPlantsSmall=spark.read.csv("dataset/nuclear_plants_small_dataset.csv", header=True,inferSchema=True)



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

# Task 4: Shuffle data into 70% training set and 30% test set

# Index labels, adding metadata (for identification) to the label column
# Fit the entire dataset, including all labels in index
labelIndexer = StringIndexer(inputCol="Status", outputCol="StatusIndexedLabel").fit(sparksNuclearPlantsSmall)

# Inputting x value
# Automatically identify categorical features, and index them seperately.
# Using maxCategories, we can select features with > 4 distinct values to be treated as continuous 
# (Continous instead of discrete, infinite number of values between two values) (Status label has 2 distinct known values currently, can be considered discrete).
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(sparksNuclearPlantsSmall)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = sparksNuclearPlantsSmall.randomSplit([0.7, 0.3])

# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Total Classified Samples), calculate sensitivity and specificity 

# Change into this
#https://spark.apache.org/docs/1.5.2/ml-decision-tree.html

# # Instance of Decision tree classifier
# treeClf = DecisionTreeClassifier(labelCol="StatusIndexedLabel", featuresCol="indexedFeatures")

# # Link the indexers and tree into a pipeline
# # Pipeline to create stages
# # Ensuring data goes through identical processing steps
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, treeClf])

# # Train model, pipeline estimator stage which produces a model which is a transformer. Running the indexer through the stages
# treeModel = pipeline.fit(trainingData)

# # Predict results, pipeline transformer stage where the model makes the predictions from the dataset
# predictions = treeModel.transform(testData)

# # Select example rows to display.
# predictions.select("prediction", "indexedLabel", "features").show(5)

# # Compute error rate
# evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
# accuracy = evaluator.evaluate(predictions)
# print ("Decision Tree Test Error = %g" % (1.0 - accuracy))

# treeModelShow = treeModel.stages[2]
# print (treeModelShow) 



# # Build the model
# svmModel = SVMWithSGD.train(trainingData, iterations=100)

# # Evaluating the model on training data
# labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
# print("Training Error = " + str(trainErr))

# # Save and load model
# svmModel.save(sc, "target/tmp/pythonSVMWithSGDModel")
# sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")





# # Instance of Support Vector Machines classifier
# svmClf = svm.SVC()
# svmClf.fit(xTrain, yTrain)
# yPredictSvmClf = svmClf.predict(xTest)
# accuracy = zero_one_loss(yTest, yPredictSvmClf)
# svmErrorRate = 1 - accuracy

# # Instance of multilayer perceptron, type of artificial neural network
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

