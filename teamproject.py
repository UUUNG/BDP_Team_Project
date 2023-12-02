from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("teamproject").getOrCreate()
	df_data = spark.read.load("airbnb_ratings_new.csv",format="csv", sep=",", inferSchmea="true", header="true")	
	feature_list = df_data.columns
	feature_list.remove("Price")
	
	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	lr = LinearRegression(featuresCol="features", labelCol="quality").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
	trainDF, testDF = df_data.randomSplit([0.8, 0.2], seed=42)
	print(trainDF.cache().count())
	print(testDF.count())
	pipeline = Pipeline(stages=[vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = PipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction", "Price")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("Price")
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName : "rmse"}))
