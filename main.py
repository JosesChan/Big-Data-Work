
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import pandas 
import seaborn
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
sparksNuclearPlantsSmall = spark.read.csv("dataset/nuclear_plants_small_dataset.csv")

#ratings = ratings.withColumn('userId', col('userId').cast('integer'))

pandasNuclearPlantsSmall = pandas.read_csv("dataset/nuclear_plants_small_dataset.csv")




# Task 1: Check to see if there are any missing values

#emptyDataRows = nuclearPlantsSmall[nuclearPlantsSmall.isnull().any(axis=1)]

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

(pandasRandomTrainNuclearPlantsSmall, pandasRandomTestNuclearPlantsSmall) = pandasNuclearPlantsSmall.randomsplit([0.7, 0.3])

xTrain = pandasRandomTrainNuclearPlantsSmall.drop(["Status"])
yTrain = pandasRandomTrainNuclearPlantsSmall["Status"]

xTest = pandasRandomTestNuclearPlantsSmall.drop(["Status"])
yTest = pandasRandomTestNuclearPlantsSmall["Status"]


# Task 5: Train a decision tree, svm and an artificial neural network. Evaluate classifiers by computing error rate (Incorrectly classified samples/Total Classified Samples), calculate sensitivity and specificity 

# Change into this
#https://spark.apache.org/docs/1.5.2/ml-decision-tree.html

# Instance of Decision tree classifier
treeClf = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
# Train tree on the data
treeClf.fit(xTrain, yTrain)
# Predict results
yPredictTreeClf = treeClf.predict(xTest)
# Compute error rate
accuracy = zero_one_loss(yTest, yPredictTreeClf)
treeErrorRate = 1 - accuracy

# Instance of Support Vector Machines classifier
svmClf = svm.SVC()
svmClf.fit(xTrain, yTrain)
yPredictSvmClf = svmClf.predict(xTest)
accuracy = zero_one_loss(yTest, yPredictSvmClf)
svmErrorRate = 1 - accuracy

# Instance of multilayer perceptron, type of artificial neural network
annClf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
annClf.fit(xTrain, yTrain)
yPredictAnnClf = annClf.predict(xTest)
accuracy = zero_one_loss(yTest, yPredictAnnClf)
annErrorRate = 1 - accuracy


# Task 6: Compare results based on task 5, which is best

print("Error rate for decision tree" + treeErrorRate)
print("Error rate for support vector machines" + svmErrorRate)
print("Error rate for artificial neural network" + annErrorRate)

# Task 7: Discuss if features can detect abnormality in reactors
# Task 8: Use mapReduce in pySpark to calculate minimum, maximum and mean for every feature

