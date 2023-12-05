from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder

if __name__ == "__main__":
	spark = SparkSession.builder.appName("teamproject").getOrCreate()
	df_data = spark.read.csv("hdfs:///user/maria_dev/airbnb_ratings_new.csv", sep=",", inferSchema="true", header="true", encoding='MacRoman')
	df_data = df_data.dropna()

	indexer_property = StringIndexer(inputCol="Property type", outputCol="PropertyIndex", handleInvalid="keep")
	indexed_property = indexer_property.fit(df_data).transform(df_data)

	indexer_country = StringIndexer(inputCol="Country", outputCol="CountryIndex", handleInvalid="keep")
	indexed_country = indexer_country.fit(indexed_property).transform(indexed_property)
	
	#df_data.select("Room type").distinct().show()
	
	indexer_room = StringIndexer(inputCol="Room type", outputCol="RoomIndex", handleInvalid="keep")
	indexed_room = indexer_room.fit(indexed_country).transform(indexed_country)

	encoder_property = OneHotEncoder(inputCol="PropertyIndex", outputCol="PropertyVec")
	encoded_property = encoder_property.transform(indexed_room)

	encoder_country = OneHotEncoder(inputCol="CountryIndex", outputCol="CountryVec")
	encoded_country = encoder_country.transform(encoded_property)
	
	encoder_room = OneHotEncoder(inputCol="RoomIndex", outputCol="RoomVec")
	encoded_room = encoder_room.transform(encoded_country)
	print(encoded_room)
	selected_columns = ["Accommodates", "Bathrooms", "Bedrooms", "Price"]
	combined_data = encoded_room.select(selected_columns + ["PropertyVec", "CountryVec", "RoomVec"])
	feature_list = combined_data.columns
	feature_list.remove("Price")
	
	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	lr = LinearRegression(featuresCol="features", labelCol="Price").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

	trainDF, testDF = combined_data.randomSplit([0.8, 0.2], seed=42)
	print(trainDF.cache().count())
	print(testDF.count())

	pipeline = Pipeline(stages=[vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction", "Price")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("Price")
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName : "rmse"}))
