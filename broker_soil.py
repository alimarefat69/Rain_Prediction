import logging
import asyncio
#from tasks import send_to_db
from amqtt.client import MQTTClient, ClientException
from amqtt.mqtt.constants import QOS_1
import mysql.connector
import json
logger = logging.getLogger(__name__)
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import datetime

#config = {
#    'listeners':{
#        'default' : {
#            'type':'tcp',
#            'bind': '0.0.0.0:1883', # by default 1883 for mqtt
#        }
#    },
#    'sys-interval':20,
#    'topic-check':{
#        'enabled':True,
#        'plugins': ['topic_taboo']

#    }
#}

#broker = Broker(config)

#@asyncio.coroutine
#def startBroker():
#    yield from broker.start()


async def brokerGetMessage():
    C = MQTTClient()
    await C.connect("mqtt://localhost:1883/")
    await C.subscribe([
        ('ali/test',QOS_1)
    ])
    logger.info("Subscribed!")
    es_client = Elasticsearch(hosts='http://localhost:9200',basic_auth=['elastic','helloworld@123'], http_compress=True)
    try:
        while True:
            message = await C.deliver_message()
            packet = message.publish_packet
            print(packet.payload.data.decode('utf-8'))
            #adding code to send data to database
            # con = mysql.connector.connect(
            #     host = 'localhost',
            #     port=3306,
            #     user = 'root',
            #     password = 'morse@gsu',
            #     db='iot',
            # )
            # cursor = con.cursor()
            val = json.loads(packet.payload.data.decode())
            # sql = f"insert into soil_sensor (datetime, Temperature, Humidity) values (\"{val['datetime']}\",{val['temperature']},{val['humidity']})"
            # print(sql)
            # cursor.execute(sql)
            # con.commit()
            # cursor.close()
            # con.close()
            # print(cursor.rowcount, 'Data Saved!')
            data = {'datetime':datetime.datetime.strftime(datetime.datetime.strptime(val['datetime'].split(".")[0], 
                                            "%Y-%m-%d %H:%M:%S"), '%Y-%m-%dT%H:%M:%S'), 'temperature':val['temperature'],'humidity':val['humidity']}
            resp = es_client.index(index='radon-data-soil', id=data['datetime'], document=data)
            print(resp)
    except ClientException as ce:
        logger.error("Client Exception: %s" %ce)

if __name__ == '__main__':
    formatter = "[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
#    asyncio.get_event_loop().run_until_complete(startBroker())
    asyncio.get_event_loop().run_until_complete(brokerGetMessage())
    asyncio.get_event_loop().run_forever()
