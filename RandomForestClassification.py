import findspark
findspark.init('/opt/spark/spark-3.3.2-bin-hadoop3')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
spark = SparkSession.builder.getOrCreate()


data = spark.read.csv('/home/amarefatvayghani/weather2.csv', header=True, inferSchema=True)
#drop a column
data = data.drop(col("Rain"))
data = data.drop(col("Time"))
data = data.withColumn("id", monotonically_increasing_id())
# data.show()
for i in range(1,50):
    data = data.withColumn("lag_temperature"+str(i), lag("Temperature", i).over(Window.partitionBy().orderBy("id")))
    data = data.withColumn("lag_hum"+str(i), lag("Humidity", i).over(Window.partitionBy().orderBy("id")))

data = data.drop(col("id"))

data = data.withColumn('Rain', col('Rain bool'))
data = data.drop(col("Rain bool"))
data = data.na.fill(0)
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data)
train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)
algo = RandomForestClassifier(featuresCol='features', labelCol='Rain')
model = algo.fit(train_data)
predictions = model.transform(test_data)
predictions.select(['Rain','prediction', 'probability']).show()
evaluator = BinaryClassificationEvaluator(labelCol='Rain', metricName='areaUnderROC')
print(evaluator.evaluate(predictions))
assembler.save('/home/amarefatvayghani/assembler')
model.save('/home/amarefatvayghani/model')