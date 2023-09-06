import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tm.Clustering.ClusteringModel import ClusteringModel

class MeanShiftClusteringModel(ClusteringModel):

	file_model_png = "tmp/MeanShiftClusteringModel.png"
	bandwidth = 2
	def __init__(self,x,selectedTrainModelFields,imageContainer):
		super().__init__(x,selectedTrainModelFields,imageContainer)
		self.model = MeanShift()
		
	def fit(self,cfg_json_string):
		#to split data for training and testing 
		self.setConfiguration(cfg_json_string)
		self.model.bandwidth = self.bandwidth
		
		#print(self.displayChart)
		if self.displayChart > 0:
			self.preLoadImage()
		else:
			self.clearModelImageResult()
			
		totalRecords = self.x.shape[0]
		totalTrainRecords = int((totalRecords*self.amountOfTrainingDataSet)/100)
		totalTestRecords = int((totalRecords*self.amountOfTestingDataSet)/100)
		
		if self.amountOfTrainingDataSet == 100:
			self.x_train = self.x
		else:
			# Split the data into training/testing sets
			self.x_train = self.x[:totalTrainRecords]
			
		if self.amountOfTestingDataSet == 100:
			self.x_test = self.x
		else:
			# Split the targets into training/testing sets
			self.x_test = self.x[:totalTestRecords]			
		#print(self.x_test)

		#the part for fitting model
		self.model.fit(self.x_train)
		self.x_pred = self.model.predict(self.x_test)
		
		self.labels = self.model.labels_
		if self.amountOfTestingDataSet < 100:
			self.labels = self.model.labels_[:totalTestRecords]
			
		str_result = "Training success"
		str_result =  "Accuracy:"+str(metrics.accuracy_score(self.labels, self.x_pred))+"\n\n"
		str_result =  str_result + "Confusion Matrix\n"
		conf_matix = confusion_matrix(self.labels, self.x_pred)	
		str_result =  str_result + str(conf_matix)
		
		return str_result
	def getConfiguration(self):
		cfg_json_str = '{"amountOfTrainingDataSet":"'+str(self.amountOfTrainingDataSet)+'","amountOfTestingDataSet":"'+str(self.amountOfTestingDataSet)+'","bandwidth":"'+str(self.bandwidth)+'","displayChart":"'+str(self.displayChart)+'"}'
		return cfg_json_str
	
	def setConfiguration(self,cfg_json_str):
		cfg_json = json.loads(cfg_json_str)
		self.amountOfTrainingDataSet = int(cfg_json["amountOfTrainingDataSet"])
		self.amountOfTestingDataSet = int(cfg_json["amountOfTestingDataSet"])
		self.bandwidth = int(cfg_json["bandwidth"])
		self.displayChart = int(cfg_json["displayChart"])
		
	def savePredictExample(self,modelFileName):
		example_header = "This model is used to predict the clusters using "+str(self.selectedTrainModelFields)+"\n"
		example_header = example_header + "You can prepare your data using the format below and save them in csv format.\n"
		example_header = example_header + "=====================================\n\n"
		example_string = ""
		
		#print(example_header)
	
		for i in range(0,len(self.selectedTrainModelFields)):
			example_string = example_string + self.selectedTrainModelFields[i]+","
		example_string = example_string.strip(',') + "\n"
		#print(example_string)
		
		for i in range(0,len(self.selectedTrainModelFields)):
			example_string = example_string + str(self.x_test[0,i])+","
		example_string = example_string.strip(',') + "\n"
		#print(example_string)
					
		outputReadme = example_header+ example_string + "\n\n"
		#outputReadme = outputReadme + "Model\n"
		
		f = open(modelFileName+"_Readme.txt", "w")
		f.write(outputReadme)
		f.close()
		
		outputExampleData = example_string	
		f = open(modelFileName+"_example.csv", "w")
		f.write(outputExampleData)
		f.close()
		
		model_str_header = ""
		fields = self.selectedTrainModelFields
		fields.append("Label")
		for i in range(0,len(fields)):
			model_str_header = model_str_header +fields[i]+","
		model_str_header = model_str_header.strip(',') + "\n"
		
		model_str_result = ""
		for i in range(0,self.x_train.shape[0]):
			for j in range(0,self.x_train.shape[1]):
				model_str_result = model_str_result +str(self.x_train[i,j])+","
			model_str_result = model_str_result +str(self.model.labels_[i])+","
			model_str_result = model_str_result.strip(',') + "\n"
		
		model_trained_result = model_str_header+model_str_result
		#print(model_trained_result)

		f = open(modelFileName+"_trained_result.csv", "w")
		f.write(model_trained_result)
		f.close()
	
	def drawFitModel(self):
		df = pd.DataFrame(self.x_train,columns=self.selectedTrainModelFields)
		df['Label'] = self.model.labels_
		
		image_parallel_coordinates = px.parallel_coordinates(df,
			color="Label",
			dimensions=self.selectedTrainModelFields,
			color_continuous_scale=px.colors.diverging.Tealrose,
			color_continuous_midpoint=2)
			
		image_parallel_coordinates.write_image(self.file_model_png)
		self.displayModelChart()
		
