from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import (
		StructType, 
		StructField, 
		StringType, 
		IntegerType, 
		DoubleType,
		)
import unicodedata
from pyspark.sql.functions import col, udf

def remove_non_ascii(text):
	return ''.join([i if ord(i) < 128 else ' ' for i in text])

if __name__ == "__main__":
	schema = StructType([
		StructField("Country", StringType(), False),
		StructField("Property type", StringType(), False),
		StructField("Accommodates", IntegerType(), False),
		StructField("Bathrooms", DoubleType(), False),
		StructField("Bedrooms", DoubleType(), False),
		StructField("Review Scores Location", DoubleType(), False),
		StructField("Price", DoubleType(), False),
		])
	spark = SparkSession.builder.appName("teamproject").getOrCreate()
	udf_remove_non_ascii = udf(remove_non_ascii, StringType())


	df_data = spark.read.load("hdfs:///user/maria_dev/airbnb_ratings_new.csv",format="csv", sep=",", inferSchema="true", header="true")
	string_columns = [col_name for col_name, col_type in df_data.dtypes if col_type == 'string'] 
	for col_name in string_columns:
		df_data = df_data.withColumn(col_name, udf_remove_non_ascii(col(col_name)))
	df_data.show(40)
	selected_columns = df_data.select("Country", "Property type", "Accommodates", "Bathrooms", "Bedrooms", "Review Scores Location", "Price")
	feature_list = selected_columns.columns
	feature_list.remove("Price")

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	lr = LinearRegression(featuresCol="features", labelCol="quality").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
	trainDF, testDF = df_data.randomSplit([0.8, 0.2], seed=42)
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
