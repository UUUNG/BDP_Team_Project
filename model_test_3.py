#-*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# SparkSession 시작
spark = SparkSession.builder.appName("ModelUsage").getOrCreate()

# 모델 로드
model_path = "hdfs:///user/maria_dev/teamproject"  # 모델이 저장된 경로
loaded_model = PipelineModel.load(model_path)

# 사용자로부터 입력 받기
input_data = [(4, 1, 2, 'Apartment', 'Italy', 'Private room')]

# 입력값을 DataFrame으로 변환
columns = ['Accommodates', 'Bathrooms', 'Bedrooms', 'Property', 'Country', 'Room type']
input_df = spark.createDataFrame(input_data, columns)

# 모델에 입력하여 예측 수행
predictions = loaded_model.transform(input_df)
predictions.select("prediction").show()

