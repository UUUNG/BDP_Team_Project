#-*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
import sys

# SparkSession 시작
spark = SparkSession.builder.appName("ModelUsage").getOrCreate()

# 모델 로드
model_path = "hdfs:///user/maria_dev/teamproject"  # 모델이 저장된 경로
model = PipelineModel.load(model_path)

print("model", model)

# 사용자로부터 입력 받기
accommodates = int(sys.argv[1])
bathrooms = int(sys.argv[2])
bedrooms = int(sys.argv[3])
property_type = sys.argv[4]
country = sys.argv[5]
room_type = sys.argv[6]

# "Accommodates", "Bathrooms", "Bedrooms", "Price", "Property type", "Country", "Room type"

input_data = [(accommodates, bathrooms, bedrooms, property_type, country, room_type)]
# 입력 데이터 예시
# Accomodates, Bathrooms, Bedrooms, Room type, Property type, Country

columns = ['Accommodates', 'Bathrooms', 'Bedrooms', 'Property type', 'Country', "Room type"]
input_df = spark.createDataFrame(input_data, columns)

# StringIndexer를 사용하여 스트링 데이터를 수치형으로 변환
indexers = [
		StringIndexer(inputCol='Property type', outputCol='PropertyIndex'),
		StringIndexer(inputCol='Room type', outputCol='RoomIndex'),
		StringIndexer(inputCol='Country', outputCol='CountryIndex')
]

indexed_df = input_df

for indexer in indexers:
	indexed_df = indexer.fit(indexed_df).transform(indexed_df)

# OneHotEncoder를 사용하여 범주형 데이터를 이진 벡터로 변환
encoders = [
		OneHotEncoder(inputCol='PropertyIndex', outputCol='PropertyVec'),
		OneHotEncoder(inputCol='RoomIndex', outputCol='RoomVec'),
		OneHotEncoder(inputCol='CountryIndex', outputCol='CountryVec')
		]
# 데이터에 원-핫 인코딩 적용
encoded_df = indexed_df
for encoder in encoders:
	encoded_df = encoder.transform(encoded_df)
# Property type, Room type, Country 각각의 고유값 개수 확인
encoded_df.select('Property type', 'Room type', 'Country').distinct().show()

# 필요한 컬럼 선택 (입력 피처 + 인코딩된 벡터) 
selected_columns = ['Accommodates', 'Bathrooms', 'Bedrooms', 'PropertyVec', 'RoomVec', 'CountryVec']
final_df = encoded_df.select(selected_columns)

# 전처리된 데이터로 벡터를 만들어 모델에 입력할 형태로 변환
#assembler = VectorAssembler(inputCols=selected_columns[:-1], outputCol="assembled_features")
#transformed_df = assembler.transform(final_df)

# 입력 데이터를 DataFrame으로 변환
# 예측 수행
predictions = model.transform(final_df)
predictions.select("prediction").show()
