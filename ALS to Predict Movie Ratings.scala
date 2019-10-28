// Databricks notebook source
// MAGIC %md #Using ALS to Predict Movie Ratings

// COMMAND ----------

// MAGIC %md ######Had a chance to learn more about the use of ALS under Spark's ML section. So without waiting further, lets try to use it to predict movie ratings. In this case, I have with me a set of movie ratings data set that was available online. Lets first begin by importing the proper library and defining the dataset into a variable.

// COMMAND ----------

import org.apache.spark.ml.recommendation.ALS
val ratings = spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/movie_ratings.csv")

// COMMAND ----------

// MAGIC %md ######As you can see below, the ratings data set consists mainly of 3 columns, the userId, movieId & rating of the movie itself.

// COMMAND ----------

ratings.printSchema()

// COMMAND ----------

// MAGIC %md ######Lets first split the data into an Array consisting of both the training & test data accordingly. We'll give training & test 0.8 & 0.2 respectively for now.

// COMMAND ----------

 val Array(training,test)=ratings.randomSplit(Array(0.8,0.2))

// COMMAND ----------

// MAGIC %md ######From there, we can then proceed to create the ALS. In this case, we will set the user column, item column and rating column accordingly based on the dataset given. We then proceed to fit the training data to the als model and transform it using input from the test data that we created earlier.

// COMMAND ----------

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

// COMMAND ----------

val model = als.fit(training)

// COMMAND ----------

val predictions = model.transform(test)

// COMMAND ----------

// MAGIC %md ######After finally carrying out the predictions, we can now see how the prediction fared based on the model we just used.

// COMMAND ----------

predictions.show()

// COMMAND ----------

// MAGIC %md ######The predictions seem fairly close for some of them but theres still a difference in error between the predicted ratings and the correct ratings itself so lets determine the error difference to determine how close our predictions were using this model itself.

// COMMAND ----------

import org.apache.spark.sql.functions._
val error = predictions.select(abs($"rating"-$"prediction"))

// COMMAND ----------

error.show()


// COMMAND ----------

error.na.drop.describe().show()

// COMMAND ----------

// MAGIC %md ######Mean error difference seems to be quite high at 0.8 to me. So probably the model would need to go through additional iterations or changes if we want to make it more accurate. But at the end of the day, this project was meant for me to practice the uses of ALS in Spark so I would definitely consider it a success! Will continue learning more about Scala and its various libraries to try to create better predictive models. Thanks for your time. Lets continue to help change our world using data science!:)

// COMMAND ----------

// MAGIC %md #END OF NOTEBOOK
