import asyncio
import logging
import socket
from datetime import datetime
import random
import json
import time
from amqtt.client import MQTTClient
from amqtt.mqtt.constants import QOS_0, QOS_1, QOS_2
from amqtt.client import ConnectException
import serial
import re
import os
import sys
async def test_pub():
	client = MQTTClient()
	while True:
		try:
			print('running try block')
			ser = serial.Serial("/dev/ttyACM0",115200)
			temp = ser.readline().decode('Ascii').strip()
			if not temp.isalnum() and ',' not in temp:
				continue
			temp, humid = temp.split(',')
			typechecker = temp.replace('.','',1).isdigit()
			if not typechecker:
				continue
			temp = float(temp)
			humid = float(humid)
			datetimenow = str(datetime.now())
			header = "datetime, Temperature, Humidity"
			if not os.path.exists('/home/pi/iotproject/log_sensor.csv'):
				with open('/home/pi/iotproject/log_sensor.csv','w+') as f:
					f.write(header + '\n')
			with open('/home/pi/iotproject/log_sensor.csv','a+') as f:	
				f.write(f'{datetimenow},{temp},{humid}\n')
			print(f"datimetime: {datetimenow}, Temperature: {temp}, Humidity: {humid}")
			print('Connected successfully')
			message = {'datetime':datetimenow, 'temperature': temp, 'humidity':humid}
			MQTT_MSG = json.dumps(message).encode('utf-8')
			await client.connect('mqtt://radonmaster.eastus.cloudapp.azure.com:1883')
			if os.path.exists('/home/pi/iotproject/error_sensor.csv'):
				with open('/home/pi/iotproject/error_sensor.csv', 'r') as f:
					lines = f.readlines()
					for line in lines[1:]:
						dataline = line.strip().split(',')
						if dataline == "":
							continue
						datetimenow_error, temperature_error, humidity_error = dataline
						humidity_error = float(humidity_error)
						temperature_error = float(temperature_error)
						message_error = {'datetime':datetimenow_error, 'temperature': temperature_error,'humidity':humidity_error}
						print(message_error)
						MQTT_MSG_ERROR = json.dumps(message_error).encode('utf-8')
						await client.publish("ali/test", MQTT_MSG_ERROR,qos=QOS_1)
				os.remove('/home/pi/iotproject/error_sensor.csv')
			await client.publish("ali/test", MQTT_MSG,qos=QOS_1)
			print(MQTT_MSG)
			await client.disconnect()
		except ConnectException:
			print('running exception')
			if not os.path.exists('/home/pi/iotproject/error_sensor.csv'):
				with open('/home/pi/iotproject/error_sensor.csv','w+') as f:
					f.write(header+'\n')
			with open('/home/pi/iotproject/error_sensor.csv','a+') as f:	
				f.write(f'{datetimenow}, {temp},{humid}\n')
			print('Temperature is ', temp)
			print('Humidity is', humid)
			print('exception done')
		except KeyboardInterrupt:
			sys.exit(0)

if __name__=='__main__':
	asyncio.run(test_pub())
