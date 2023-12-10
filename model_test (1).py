#-*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel, OneHotEncoder, VectorAssembler
import sys
from pyspark.sql.functions import when

# SparkSession 시작
spark = SparkSession.builder.appName("ModelUsage").getOrCreate()

property_type_list = ['Apartment', 'Bed & Breakfast', 'House', 'Villa', 'Loft', 'Guesthouse', 'Other', 'Dorm', 'Townhouse', 'Condominium', 'Cabin', 'Bungalow', 'Boat', 'Treehouse', 'Train', 'Camper/RV', 'Castle', 'Chalet', 'Timeshare', 'Tent', 'Guest suite', 'Earth House', 'Hostel', 'Serviced apartment', 'Boutique hotel']

country_list = [
'Hong Kong', 'Italy', 'Belgium', 'Austria', 'Denmark', 'Australia', 'Spain', 'Germany', 'France', 'United Kingdom', 'China', 'Greece', 'Canada', 'United States', 'Ireland', 'Netherlands', 'Switzerland']

oom_type_list = ['Entire home/apt', 'Private room', 'Shared room']

# 모델 로드
model_path = "hdfs:///user/maria_dev/teamproject3"
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
property_indexer = StringIndexer(inputCol='Property type', outputCol='PropertyIndex', handleInvalid='keep')
country_indexer = StringIndexer(inputCol='Country', outputCol='CountryIndex', handleInvalid='keep')
room_indexer = StringIndexer(inputCol='Room type', outputCol='RoomIndex', handleInvalid='keep')

#indexed_df = input_df

# 입력값에 대해 변환 적용
indexed_df = property_indexer.fit(input_df).transform(input_df)
indexed_df = country_indexer.fit(indexed_df).transform(indexed_df)
indexed_df = room_indexer.fit(indexed_df).transform(indexed_df)

# 필요한 컬럼 선택 (입력 피처 + 인코딩된 벡터) 
selected_columns = ['Accommodates', 'Bathrooms', 'Bedrooms', 'PropertyIndex', 'CountryIndex', 'RoomIndex']

assembler = VectorAssembler(inputCols=selected_columns, outputCol="features_1")
final_df = assembler.transform(indexed_df)
#for indexer in indexers:
#	indexed_df = indexer.fit(indexed_df).transform(indexed_df)

# OneHotEncoder를 사용하여 범주형 데이터를 이진 벡터로 변환
#encoders = [
#		OneHotEncoder(inputCol='PropertyIndex', outputCol='PropertyVec'),
#		OneHotEncoder(inputCol='CountryIndex', outputCol='CountryVec'),
#		OneHotEncoder(inputCol='RoomIndex', outputCol='RoomVec'),
#		]
# 데이터에 원-핫 인코딩 적용
#encoded_df = indexed_df
#for encoder in encoders:
#	encoded_df = encoder.transform(encoded_df)
# Property type, Room type, Country 각각의 고유값 개수 확인
#encoded_df.select('Property type', 'Country', 'Room type').distinct().show()

# One-Hot Encoding을 위한 인코더 정의 및 적용
encoder_property = OneHotEncoder(inputCol='PropertyIndex', outputCol='PropertyVec')
encoded_df = encoder_property.transform(final_df)

encoder_country = OneHotEncoder(inputCol='CountryIndex', outputCol='CountryVec')
encoded_df = encoder_country.transform(encoded_df)

encoder_room = OneHotEncoder(inputCol='RoomIndex', outputCol='RoomVec')
encoded_df = encoder_room.transform(encoded_df)

#encoded_df = encoded_df.withColumn("weights", when(encoded_df["Price"] <= 100, 10).when(encoded_df["Price"] <= 200, 9).when(encoded_df["Price"] <= 300, 8).when(encoded_df["Price"] <= 400, 7).when(encoded_df["Price"] <= 500, 6).when(encoded_df["Price"] <= 600, 5).when(encoded_df["Price"] <= 700, 4).when(encoded_df["Price"] <= 800, 3).when(encoded_df["Price"] <= 900, 2).otherwise(1))
# 예측 수행
#prediction_input = encoded_df.select(encoded_df)
predictions = model.transform(encoded_df)
predictions.select("prediction").show()
