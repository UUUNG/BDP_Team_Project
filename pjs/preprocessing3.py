from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql.types import DoubleType, FloatType, IntegerType
from pyspark.sql.functions import corr
from pyspark.sql.functions import countDistinct
from pyspark.sql import functions as F


if __name__ == "__main__":
	spark = SparkSession.builder.appName("preprocessing").config("spark.debug.maxToStringFields","100").getOrCreate()
	df = spark.read.csv("/user/maria_dev/project/airbnb_ratings_new.csv", header="true", sep=",", inferSchema="true", encoding='MacRoman')

	#df.show()

	# 컬럼 명칭 출력
	print("Columns name\n", df.columns)
	print("-" * 50)

	# Row 수 출력
	print("Row count\n", df.count())
	print("-" * 50)

	# 컬럼별 null 수 출력
	df_null = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
	null_counts = df_null.collect()[0]
	print("Null count")
	for c in df.columns:
		print(f"{c}: {null_counts[c]}")
	print("-" * 50)

	# null 제거
	df_dropna = df.dropna()

	# null 제거 후 row 수 출력
	print("Row count after dropna\n", df_dropna.count())
	print("-" * 50)

	# numeric column과 price와의 상관관계
	numeric_columns = [c for c, t in df.dtypes if t in ['double', 'float', 'int']]
	print("Correlation with price")
	for c in numeric_columns:
		correlation = df_dropna.stat.corr('Price', c)
		print("Correlation with Price and {}: {}".format(c, correlation))
	print("-" * 50)

	# 컬럼별 distinct value 수
	categoric_columns = [c for c in df.columns if c not in numeric_columns]
	print("Categorical column's distinct value count")
	for c in categoric_columns:
		distinct_count = df_dropna.agg(countDistinct(col(c)).alias("distinct_count")).collect()[0]["distinct_count"]
		print("Distinct values counts {}: {}".format(c, distinct_count))
	print("-" * 50)

	df_dropna.createOrReplaceTempView("df_view")

	# country별 평균 가격
	country_prices = spark.sql("""
	SELECT Country, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY Country
	ORDER BY AveragePrice desc
	""")
	country_prices.show(country_prices.count())

	# property type별 평균 가격
	propertyType_prices = spark.sql("""
	SELECT `Property type`, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY `Property type`
	ORDER BY AveragePrice desc
	""")
	propertyType_prices.show(propertyType_prices.count())

	# room type별 평균 가격
	roomType_prices = spark.sql("""
	SELECT `Room type`, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY `Room type`
	ORDER BY AveragePrice desc
	""")
	roomType_prices.show(roomType_prices.count())


	# accommodates별 평균 가격
	accommodates_prices = spark.sql("""
	SELECT `Accommodates`, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY `Accommodates`
	ORDER BY AveragePrice desc
	""")
	accommodates_prices.show(accommodates_prices.count())

	# bathrooms별 평균 가격
	bathrooms_prices = spark.sql("""
	SELECT `Bathrooms`, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY `Bathrooms`
	ORDER BY AveragePrice desc
	""")
	bathrooms_prices.show(bathrooms_prices.count())

	# bedrooms별 평균 가격
	bedrooms_prices = spark.sql("""
	SELECT `Bedrooms`, AVG(Price) as AveragePrice
	FROM df_view
	GROUP BY `Bedrooms`
	ORDER BY AveragePrice desc
	""")
	bedrooms_prices.show(bedrooms_prices.count())

	# 컬럼 선택
	df_selected = df.select('Price', 'Country', 'Property type', 'Room type', 'Accommodates', 'Bathrooms', 'Bedrooms')
	df_selected = df_selected.dropna()
	print("Row count after column selection\n", df_selected.count())

	# price 0인 경우 제거
	df_selected.filter(df_selected["Price"] == 0).count()
	df_filtered = df_selected.filter(df_selected["Price"] != 0)
	print("Row count Price!=0\n", df_filtered.count())

	# describe Price
	df_filtered.createOrReplaceTempView("df_filtered_view")
	describe = spark.sql("""
	SELECT 
	MIN(Price) as min_price, 
	MAX(Price) as max_price, 
	AVG(Price) as avg_price,
	percentile_approx(Price, 0.5) as median_price
	FROM df_filtered_view
	""")
	describe.show()

	# country별 count
	country_count = spark.sql("""
	SELECT Country, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY Country
	ORDER BY Count desc
	""")
	country_count.show(country_count.count())

	# property type별 count
	propertyType_count = spark.sql("""
	SELECT `Property type`, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY `Property type`
	ORDER BY Count desc
	""")
	propertyType_count.show(propertyType_count.count())

	# room type별 count
	roomType_count = spark.sql("""
	SELECT `Room type`, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY `Room type`
	ORDER BY Count desc
	""")
	roomType_count.show(roomType_count.count())

	# accommodates별 count
	accommodates_count = spark.sql("""
	SELECT `Accommodates`, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY `Accommodates`
	ORDER BY Count desc
	""")
	accommodates_count.show(accommodates_count.count())

	# bathrooms별 count
	bathrooms_count = spark.sql("""
	SELECT `Bathrooms`, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY `Bathrooms`
	ORDER BY Count desc
	""")
	bathrooms_count.show(bathrooms_count.count())

	# bedrooms별 count
	bedrooms_count = spark.sql("""
	SELECT `Bedrooms`, COUNT(Price) as Count
	FROM df_filtered_view
	GROUP BY `Bedrooms`
	ORDER BY Count desc
	""")
	bedrooms_count.show(bedrooms_count.count())

	# count 30이상 필터링
	country_count = df_filtered.groupBy("Country") \
	.agg(F.count("Price").alias("Count")) \
	.filter(F.col("Count") >= 30)

	propertyType_count = df_filtered.groupBy("Property type")\
	.agg(F.count("Price").alias("Count"))\
	.filter(F.col("Count") >= 30)

	roomType_count = df_filtered.groupBy("Room type")\
	.agg(F.count("Price").alias("Count"))\
	.filter(F.col("Count") >= 30)

	accommodates_count = df_filtered.groupBy("Accommodates")\
	.agg(F.count("Price").alias("Count"))\
	.filter(F.col("Count") >= 30)

	bathrooms_count = df_filtered.groupBy("Bathrooms")\
	.agg(F.count("Price").alias("Count"))\
	.filter(F.col("Count") >= 30)

	bedrooms_count = df_filtered.groupBy("Bedrooms")\
	.agg(F.count("Price").alias("Count"))\
	.filter(F.col("Count") >= 30)

	df_filtered = df_filtered.join(country_count, "Country")\
	.join(propertyType_count, "Property type")\
	.join(roomType_count, "Room type")\
	.join(accommodates_count, "Accommodates")\
	.join(bathrooms_count, "Bathrooms")\
	.join(bedrooms_count, "Bedrooms")

	df_filtered = df_filtered.select('Price', 'Country', 'Property type', 'Room type', 'Accommodates', 'Bathrooms', 'Bedrooms')

	df_filtered.createOrReplaceTempView("df_filtered_view")

	# 가격 구간별 count
	price_count = spark.sql("""
	SELECT CASE 
			WHEN price BETWEEN 1 AND 100 THEN '1-100'
			WHEN price BETWEEN 101 AND 200 THEN '101-200'
			WHEN price BETWEEN 201 AND 300 THEN '201-300'
			WHEN price BETWEEN 301 AND 400 THEN '301-400'
			WHEN price BETWEEN 401 AND 500 THEN '401-500'
			WHEN price BETWEEN 501 AND 600 THEN '501-600'
			WHEN price BETWEEN 601 AND 700 THEN '601-700'
			WHEN price BETWEEN 701 AND 800 THEN '701-800'
			WHEN price BETWEEN 801 AND 900 THEN '801-900'
			WHEN price BETWEEN 901 AND 999 THEN '901-999'
			END AS Price_Range, 
			COUNT(*) AS Count
	FROM df_filtered_view
	GROUP BY Price_Range
	ORDER BY Count desc
	""")
	price_count.show(price_count.count())

	df_filtered.write.csv("hdfs:///user/maria_dev/project/airbnb.csv", header=True)
