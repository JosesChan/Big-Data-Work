
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import findspark
import pandas 

findspark.init()

spark = SparkSession.builder.getOrCreate()

# Read csv file into spark
ratings = spark.read.csv("dataset/ratings.csv",header=True)
movies = spark.read.csv("dataset/movies.csv",header=True)

#Create dataframe with columns
ratings = ratings.withColumn('userId', col('userId').cast('integer')). withColumn('movieId',col('movieId').cast('integer')). withColumn('rating', col('rating').cast('float')).drop('timestamp')
ratings.show(5)

# Equijoin the movieId column in ratings and movies
userRatings = ratings.join(movies, on='movieId')
# Create dataframe with Pandas
userRatings = userRatings.toPandas()
# Create pivot table from dataframe
userRatings = userRatings.pivot_table(index=['userId'], columns=['title'], values='rating')
# Show first 5 rows
userRatings.head()

# Return tuple of the array dimensions in dataset of 610 users and 9719 movies
userRatings.shape

# Set neighbour amount for k-nearest neighbour amount
kNeighbors = 10

# Pearson implementation
# Calculate similarity using Pearson Correlation Coefficient
userDistancePearson = userRatings.transpose().corr(method='pearson')

# Keep rows and columns that have atleast k+1 neighbors and values that are non-NA.
userDistancePearson = userDistancePearson.dropna(axis=0, thresh=kNeighbors+1).dropna(axis=1,
thresh=kNeighbors+1)

# Create square matrix, remove additional rows 
userDistancePearson = userDistancePearson.loc[userDistancePearson.columns]

# Calculate distance between users
# Pearson Correlation Coefficient boundaries, 1 to -1, Pearson distance metric boundaries 0 to 2, Pearson distance = 1 - r
userDistancePearson = 1.0 - userDistancePearson
# account for floating point precision problem
userDistancePearson[userDistancePearson < 0] = 0

# k-nn forming neighborhoods 
# metric set to precomputed, X is assumed to be a distance matrix and must be square during fit
modelKNN = NearestNeighbors(metric='precomputed', algorithm='brute', n_neighbors=kNeighbors, n_jobs=-1)
# fitting of X, X graph might be sparsely populated making non-zero elements the only canidates for being a neighbour
modelKNN.fit(csr_matrix(userDistancePearson.fillna(0).values))
# predict with k-nn model
similarity, indexes = modelKNN.kNeighbors(csr_matrix(userDistancePearson.fillna(0).values),n_neighbors=kNeighbors)

# average top 10 ratings using pearson
# get neighbor id from user distance pearson dataframe as it currently stores the int index and not the actual id
# then use K-nn imputation, calculates missing ratings
# then calculate average top 10 from list
neighborhoods_pearson = pandas.DataFrame({'neighborhood_ids':[userDistancePearson.iloc[neighbors].index.to_list() for neighbors in indexes],'distance': similarity.tolist()},index=userDistancePearson.index)
user1NeighborsPearson = neighborhoods_pearson['neighborhood_ids'].loc[1]
userRatings.loc[user1NeighborsPearson].mean().sort_values(ascending=False)[:10]

# Spearman's rank correlation implementation
# Calculate similarity using Pearson Correlation Coefficient
userDistancesSpearman = userRatings.transpose().corr(method='spearman')

# Keep rows and columns that have atleast k+1 neighbors and values that are non-NA.
userDistancesSpearman = userDistancesSpearman.dropna(axis=0, thresh=kNeighbors+1).dropna(axis=1,thresh=kNeighbors+1)
# Create square matrix, remove additional rows 
userDistancesSpearman = userDistancesSpearman.loc[userDistancesSpearman.columns]

# Calculate distance between users
userDistancesSpearman = 1.0 - userDistancesSpearman
userDistancesSpearman[userDistancesSpearman < 0] = 0

# k-nn forming neighborhoods
modelKNN = NearestNeighbors(metric='precomputed', algorithm='brute', n_neighbors=kNeighbors, n_jobs=-1)
modelKNN.fit(csr_matrix(userDistancesSpearman.fillna(0).values))
similarity, indexes = modelKNN.kNeighbors(csr_matrix(userDistancesSpearman.fillna(0).values),n_neighbors=kNeighbors)

# average ratings using spearman's
neighborhoods_spearman = pandas.DataFrame({'neighborhood_ids':[userDistancesSpearman.iloc[neighbors].index.to_list() for neighbors in indexes],'distance': similarity.tolist()}, index=userDistancesSpearman.index)
user1NeighborsSpearman = neighborhoods_spearman['neighborhood_ids'].loc[1]
userRatings.loc[user1NeighborsSpearman].mean().sort_values(ascending=False)[:10]

#Both spearman and pearson recommendations examined through breakpoints
#movies recommendation to user 1 based neighborhoods with pearson
userRatings.loc[user1NeighborsPearson].mean().sort_values(ascending=False)[:10]
#movies recommendation to user 1 based neighborhoods with spearman
userRatings.loc[user1NeighborsSpearman].mean().sort_values(ascending=False)[:10]


