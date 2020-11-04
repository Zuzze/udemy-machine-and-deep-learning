from __future__ import print_function

# new MLLib API
from pyspark.ml.regression import LinearRegression

# in Spark 2, you have to use SparkSession instead of SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    # Windows only: "spark.sql.warehouse.dir", "file:///C:/temp"
    spark = SparkSession.builder.config("spark.sql.warehouse.dir").appName("LinearRegression").getOrCreate()

    # Load up our data and convert it to the format MLLib expects.
    inputLines = spark.sparkContext.textFile("regression.txt")
    # MLLib requires dense Vector array
    data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

    # Convert this RDD to a DataFrame
    colNames = ["label", "features"]
    df = data.toDF(colNames)

    # Note, there are lots of cases where you can avoid going from an RDD to a DataFrame.
    # Perhaps you're importing data from a real database. Or you are using structured streaming
    # to get your data.

    # Let's split our data into training data and testing data
    trainTest = df.randomSplit([0.5, 0.5])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # Now create our linear regression model
    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Train the model using our training data
    model = lir.fit(trainingDF)

    # Now see if we can predict values in our test data.
    # Generate predictions using our linear regression model for all features in our
    # test dataframe:
    fullPredictions = model.transform(testDF).cache()

    # Extract the predictions and the "known" correct labels.
    predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
    labels = fullPredictions.select("label").rdd.map(lambda x: x[0])

    # Zip them together
    predictionAndLabel = predictions.zip(labels).collect()

    # Print out the predicted and actual values for each point
    for prediction in predictionAndLabel:
      print(prediction)


    # Stop the session
    spark.stop()


    '''
    Result: predicted value, actual value
    (-2.645237711750381, -3.74)
    (-1.8099204346917857, -2.58)
    (-1.6824991551404747, -2.54)
    (-1.6541833152401832, -2.29)
    (-1.3356301163619053, -2.12)
    (-1.3922617961624881, -1.96)
    (-1.3922617961624881, -1.94)
    (-1.2860773965363954, -1.91)
    (-1.307314276461614, -1.91)
    (-1.3285511563868324, -1.88)
    (-1.3922617961624881, -1.87)
    (-1.2931563565114683, -1.8)
    (-1.1869719568853756, -1.75)
    (-1.1657350769601569, -1.74)
    (-1.0241558774587, -1.67)
    (-1.1444981970349384, -1.65)
    (-1.2931563565114683, -1.64)
    (-1.1374192370598657, -1.59)
    (-1.0170769174836274, -1.58)
    (-1.0949454772094287, -1.57)
    (-0.9675241976581173, -1.48)
    (-1.0241558774587, -1.47)
    (-1.1161823571346472, -1.42)
    (-0.9887610775833358, -1.36)
    (-0.9108925178575346, -1.34)
    (-0.7976291582563689, -1.3)
    '''
