# Author: Alex Aw (last edit 06Apr2022)
# Objective: Spark script for training ALS model on AWS EMR via AWS Data Pipeline.

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString

spark = SparkSession.builder.appName('BeerRecSys').getOrCreate()

### load file
file = "s3://bucket/beer_reviews_date.csv" #1.5m reviews
ratings_full = spark.read.csv(file, header=True)

# mapping new column names and converting ratings to float
ratings_spdf = ratings_full.select('review_profilename', 'beer_name', 'review_overall')
newcolnames = ['userid', 'itemid', 'rating']
ratings_spdf = ratings_spdf.toDF(*newcolnames)
ratings_spdf = ratings_spdf.withColumn("rating", ratings_spdf.rating.cast("Float"))

### Indexing UserID and ItemID
userIndexer = StringIndexer(inputCol='userid', outputCol='userIndex').fit(ratings_spdf)
itemIndexer = StringIndexer(inputCol='itemid', outputCol='itemIndex').fit(ratings_spdf)

pipeline = Pipeline(stages=[userIndexer, itemIndexer])
indexedRatings = pipeline.fit(ratings_spdf).transform(ratings_spdf)

### Training model
## test, train, split
(training, test) = indexedRatings.randomSplit([0.7, 0.3])

## train model
als = ALS(maxIter=20, rank=40, regParam=0.25, userCol="userIndex", itemCol="itemIndex", ratingCol="rating", coldStartStrategy="drop", implicitPrefs=False)
model = als.fit(training)

## evaluate model
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Square Error = ", rmse)

### Recommender List
userRecs = model.recommendForAllUsers(10)
beerRecs = model.recommendForAllItems(10)

### flatten DF
flatUserRecs = userRecs.withColumn("itemAndRating", explode(userRecs.recommendations)).select("userIndex", "itemAndRating.*")

### map back user and item names
userConverter = IndexToString(inputCol="userIndex", outputCol="userid", labels=userIndexer.labels)
itemConverter = IndexToString(inputCol="itemIndex", outputCol="itemid", labels=itemIndexer.labels)

convertedUserRecs = Pipeline(stages=[userConverter, itemConverter]).fit(indexedRatings).transform(flatUserRecs)
CUR = convertedUserRecs.select("userid", "itemid", "rating")


### saving the file to s3
CUR.coalesce(10).write.mode("overwrite").csv("s3://bucket/usertop10rec.csv")

