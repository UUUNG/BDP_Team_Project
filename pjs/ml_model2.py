from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import log, exp
from pyspark.sql.functions import when

if __name__ == "__main__":
	spark = SparkSession.builder.appName("teamproject").getOrCreate()
	df_data = spark.read.csv("hdfs:///user/maria_dev/project/airbnb.csv", sep=",", inferSchema="true", header="true", encoding='MacRoman')

	indexer_property = StringIndexer(inputCol="Property type", outputCol="PropertyIndex", handleInvalid="keep")
	indexed_property = indexer_property.fit(df_data).transform(df_data)

	indexer_country = StringIndexer(inputCol="Country", outputCol="CountryIndex", handleInvalid="keep")
	indexed_country = indexer_country.fit(indexed_property).transform(indexed_property)
		
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
	
	# 가중치
	combined_data = combined_data.withColumn(
			"weights",
			when(combined_data["Price"] <= 100, 10)
			.when(combined_data["Price"] <= 200, 9)
			.when(combined_data["Price"] <= 300, 8)
			.when(combined_data["Price"] <= 400, 7)
			.when(combined_data["Price"] <= 400, 6)
			.when(combined_data["Price"] <= 600, 5)
			.when(combined_data["Price"] <= 700, 4)
			.when(combined_data["Price"] <= 800, 3)
			.when(combined_data["Price"] <= 900, 2)
			.otherwise(1)
			)	

	feature_list = combined_data.columns

	feature_list.remove("Price")
					
	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	lr = LinearRegression(featuresCol="features", labelCol="Price", weightCol="weights").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

	trainDF, testDF = combined_data.randomSplit([0.8, 0.2], seed=42)
	print(trainDF.cache().count())
	print(testDF.count())

	pipeline = Pipeline(stages=[vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	print("\npredDF:", predDF)

	predAndLabel = predDF.select("prediction", "Price")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("Price")
	print("r2: ", evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print("mae: ", evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print("rmse: ", evaluator.evaluate(predAndLabel, {evaluator.metricName : "rmse"}))
