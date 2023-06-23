On raspberrypi run the "pub_soil_temp".
this code is the publisher to get the data from Arduino and send to the Broker on the cloud.

On the Azure cloud VM run the following codes:
"broker_soil" retrieve the data from "pub_soil_temp". The humidity and temperature data will be sent to the dashboard for visualization as well as the "inferencing".
"RandomForestClassification" performs the training.
"inferencing" does the inferencing based on the trained model on the real data to come up with the prediction and probability on the visulization graphs on the dashboard.
