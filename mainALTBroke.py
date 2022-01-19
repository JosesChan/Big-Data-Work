#import pyspark

#from pyspark.sql import SparkSession

#spark = SparkSession.builder.getOrCreate()

#df = spark.sql("select ‘spark’ as hello")
#df.show()

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import findspark
import pandas as pd

findspark.init()

spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
ratings = spark.read.csv("dataset/ratings.csv",header=True)
movies = spark.read.csv("dataset/movies.csv",header=True)

#Create dataframe with columns
ratings = ratings.withColumn('userId', col('userId').cast('integer')). withColumn('movieId',col('movieId').cast('integer')). withColumn('rating', col('rating').cast('float')).drop('timestamp')
ratings.show(5)

# Equijoin the movieId column in ratings and movies
user_ratings = ratings.join(movies, on='movieId')
# Create dataframe with Pandas
user_ratings = user_ratings.toPandas()
# Create pivot table from dataframe
user_ratings = user_ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
# Show first 5 rows
user_ratings.head()

# Return tuple of the array dimensions in dataset of 610 users and 9719 movies
user_ratings.shape

# Calculate similarity using Pearson Correlation Coefficient
user_distances_pearson = user_ratings.transpose().corr(method='pearson')
user_distances_pearson

# Set neighbour amount for k-nearest neighbour amount
kneighbors = 10

# Keep rows and columns with atleast kneighbors+1 number and non-NA values.
user_distances_pearson = user_distances_pearson.dropna(axis=0, thresh=kneighbors+1).dropna(axis=1,
thresh=kneighbors+1)

# Make the matrix square (dropping the extra rows as well)
user_distances_pearson = user_distances_pearson.loc[user_distances_pearson.columns]

#  Pearson Correlation Coefficient boundaries, 1 to -1, Pearson distance boundaries 0 to 2, Pearson distance = 1 - r
user_distances_pearson = 1.0 - user_distances_pearson
# floating point precision problem
user_distances_pearson[user_distances_pearson < 0] = 0


# k-nn model
# metric is set to precomputed instead of the default metric minkowski, 
# X is assumed to be a distance matrix and must be square during fit
model_knn = NearestNeighbors(metric='precomputed', algorithm='brute', n_neighbors=kneighbors, n_jobs=-1)
# fitting of X
# X graph might be sparsely populated, 
# making non-zero elements the only canidates for being a neighbour
model_knn.fit(csr_matrix(user_distances_pearson.fillna(0).values))
# create list of similarity for predictions
similarity, indexes = model_knn.kneighbors(csr_matrix(user_distances_pearson.fillna(0).values),n_neighbors=kneighbors)


# dataframe with list of nearest neighbours per user and associated distances
neighborhoods_pearson = pd.DataFrame({'neighborhood_ids':[user_distances_pearson.iloc[neighbors].index.to_list() for neighbors in indexes],'distance': similarity.tolist()},index=user_distances_pearson.index)

neighborhoods_pearson

user_1_neighbors_pearson = neighborhoods_pearson['neighborhood_ids'].loc[1]
user_1_neighbors_pearson

user_ratings.loc[user_1_neighbors_pearson].mean().sort_values(ascending=False)[:10]

# Spearman's rank correlation implementation
user_distances_spearman = user_ratings.transpose().corr(method='spearman')
user_distances_spearman

kneighbors = 10

# Keep only the rows+columns with at least kneighbors+1 number non-NA values.
user_distances_spearman = user_distances_spearman.dropna(axis=0, thresh=kneighbors+1).dropna(axis=1,thresh=kneighbors+1)
# Make the matrix square (dropping the extra rows as well)
user_distances_spearman = user_distances_spearman.loc[user_distances_spearman.columns]

user_distances_spearman = 1.0 - user_distances_spearman
user_distances_spearman[user_distances_spearman < 0] = 0


model_knn = NearestNeighbors(metric='precomputed', algorithm='brute', n_neighbors=kneighbors, n_jobs=-1)
model_knn.fit(csr_matrix(user_distances_spearman.fillna(0).values))
similarity, indexes = model_knn.kneighbors(csr_matrix(user_distances_spearman.fillna(0).values),n_neighbors=kneighbors)


neighborhoods_spearman = pd.DataFrame({'neighborhood_ids':[user_distances_spearman.iloc[neighbors].index.to_list() for neighbors in indexes],'distance': similarity.tolist()},
index=user_distances_spearman.index)
neighborhoods_spearman


user_1_neighbors_spearman = neighborhoods_spearman['neighborhood_ids'].loc[1]
user_1_neighbors_spearman

user_ratings.loc[user_1_neighbors_spearman].mean().sort_values(ascending=False)[:10]


#Compare both spearman and pearson recommendations
#User 1 neighborhoods according to spearman
user_1_neighbors_spearman
#movies recommendation to user 1 based neighborhoods with spearman
user_ratings.loc[user_1_neighbors_spearman].mean().sort_values(ascending=False)[:10]

#User 1 neighborhoods according to pearson
user_1_neighbors_pearson
#movies recommendation to user 1 based neighborhoods with pearson
user_ratings.loc[user_1_neighbors_pearson].mean().sort_values(ascending=False)[:10]