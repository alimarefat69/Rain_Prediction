import logging
import asyncio
#from tasks import send_to_db
from amqtt.client import MQTTClient, ClientException
from amqtt.mqtt.constants import QOS_1
# import mysql.connector
import json
logger = logging.getLogger(__name__)
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import datetime
import findspark
findspark.init('/opt/spark/spark-3.3.2-bin-hadoop3')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import  RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import PipelineModel

spark = SparkSession.builder.getOrCreate()
async def brokerGetMessage():
    C = MQTTClient()
    es_client = Elasticsearch(hosts='http://radonmaster.eastus.cloudapp.azure.com:9200',basic_auth=['elastic','helloworld@123'], http_compress=True)
    try:
        data_counter = 0
        while True:
            await C.connect("mqtt://radonmaster.eastus.cloudapp.azure.com:1883/")
            await C.subscribe([
        ('ali/test',QOS_1)
    ])
            logger.info("Subscribed!")
            message = await C.deliver_message()
            if message is None:
                continue
            if message.publish_packet is None:
                continue
            packet = message.publish_packet
            if packet is None:
                continue
            print(packet.payload.data.decode('utf-8'))
            val = json.loads(packet.payload.data.decode())
            #disconnect from the broker
            await C.disconnect()
            data = {'Time':datetime.datetime.strftime(datetime.datetime.strptime(val['datetime'].split(".")[0], "%Y-%m-%d %H:%M:%S"), '%Y-%m-%dT%H:%M:%S'),'Temperature':val['temperature'],'Humidity':val['humidity']}
            data_counter += 1
            if data_counter == 1:
                df = spark.createDataFrame([(data_counter,data['Time'], data['Temperature'], data['Humidity'])], ['id','Time', 'Temperature', 'Humidity'])
            else:
                df = df.union(spark.createDataFrame([( data_counter,data['Time'], data['Temperature'], data['Humidity'])], ['id','Time', 'Temperature', 'Humidity']))
            if data_counter % 2 == 0:
                df1 = df.alias("df1")
                df1 = df1.drop(col("Time"))
                for i in range(1,50):
                    df1 = df1.withColumn("lag_temperature"+str(i), lag("Temperature", i).over(Window.partitionBy().orderBy("id")))
                    df1 = df1.withColumn("lag_hum"+str(i), lag("Humidity", i).over(Window.partitionBy().orderBy("id")))
                df1 = df1.drop(col("id"))
                df1 = df1.na.fill(0)
                assembler = VectorAssembler.load('/home/amarefatvayghani/assembler')
                df1 = assembler.transform(df1)
                model = RandomForestClassificationModel.load('/home/amarefatvayghani/model')
                predictions = model.transform(df1)
                predictions.select(['prediction', 'probability']).show()
                #get the perdiction of the last row
                last_row = predictions.collect()[-1]
                last_row = last_row.asDict()
                print(last_row)
                # data['prediction'] = last_row['prediction']
                #get the probability of the last row prediction positive class
                data['probability'] = last_row['probability'][1]
                data['prediction'] = 1 if last_row['probability'][1] >= 0.2 else 0
                resp = es_client.index(index='cloud-prediction', id=data['Time'], document=data)
                print(resp)
    except ClientException as ce:
        logger.error("Client Exception: %s" %ce)

if __name__ == '__main__':
    formatter = "[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
#    asyncio.get_event_loop().run_until_complete(startBroker())
    asyncio.get_event_loop().run_until_complete(brokerGetMessage())
    asyncio.get_event_loop().run_forever()