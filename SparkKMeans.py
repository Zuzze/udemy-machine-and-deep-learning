from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

# define number of means used
K = 5

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    # create bunch of random centroids
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X) #numpy
    return X

random.seed(0)

# Load the data; note I am normalizing it with scale() - very important!
# parallelize transforms data to RDD (resilient distributed dataset)
data = sc.parallelize(scale(createClusteredData(100, K)))

# Build the model (cluster the data) from MLLib
# runs=10 was depracated
clusters = KMeans.train(data, K, maxIterations=10, initializationMode="random")

# Print out the cluster assignments
# if you have a call having more than 1 function, have to be cached so that won't be computed more than once
resultRDD = data.map(lambda point: clusters.predict(point)).cache()

print("Counts by value:")
counts = resultRDD.countByValue()
#prints defaultdict(<class 'int'>, {3: 20, 1: 40, 2: 10, 0: 10, 4: 20})
print(counts)

print("Cluster assignments:")
results = resultRDD.collect()
# prints [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]
print(results)


# Evaluate clustering by computing Within Set Sum of Squared Errors (SSSE)
# computed squared error for each point
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# combine 2 rows and add them to RDD
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))
# prints 33.29648521997769


# Things to try:
# What happens to WSSSE as you increase or decrease K? Why?
# What happens if you don't normalize the input data before clustering?
# What happens if you change the maxIterations or runs parameters?
