import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
import cv2
from gui.ImageModelConfiguration import ImageModelConfiguration

class ClusteringModel:
	displayChart = 1
	figure_width = 3.0
	figure_height = 1.7
	figure_canvas = None
	selectedTrainModelFields = []
	
	amountOfTrainingDataSet = 100
	amountOfTestingDataSet = 100
	n_clusters = 3
	x = []

	def __init__(self,x,selectedTrainModelFields,imageContainer):
		
		a = np.array(x)
		#b = np.array(y)
		a = a.astype(np.float)
		#b = b.astype(np.float)
		
		self.x = a.transpose()		
		self.selectedTrainModelFields = selectedTrainModelFields

		imputer = SimpleImputer(missing_values=np.nan,strategy = "mean") 
		imputer.fit(self.x)
		self.x = imputer.transform(self.x)
		self.imageContainer = imageContainer
		
	def fit():
		print("to do more ...")
		
	def predict(self,x):
		p_data = np.array(x).astype(np.float).transpose()
		answers = self.model.predict(p_data)
		return answers
		
	def getConfiguration(self):
		cfg_json_str = '{"amountOfTrainingDataSet":"'+str(self.amountOfTrainingDataSet)+'","amountOfTestingDataSet":"'+str(self.amountOfTestingDataSet)+'","n_clusters":"'+str(self.n_clusters)+'","displayChart":"'+str(self.displayChart)+'"}'
		return cfg_json_str
	
	def setConfiguration(self,cfg_json_str):
		cfg_json = json.loads(cfg_json_str)
		self.amountOfTrainingDataSet = int(cfg_json["amountOfTrainingDataSet"])
		self.amountOfTestingDataSet = int(cfg_json["amountOfTestingDataSet"])
		self.n_clusters = int(cfg_json["n_clusters"])
		self.displayChart = int(cfg_json["displayChart"])

	def preLoadImage(self):
		original_image = cv2.imread("tmp/loading.png", 1)
		scaled_image = cv2.resize(original_image, (ImageModelConfiguration.width,ImageModelConfiguration.height))  
		
		success, encoded_image = cv2.imencode('.png', scaled_image)
		imgbytes = encoded_image.tobytes()
		self.imageContainer.update(data=imgbytes)

	def displayModelChart(self):
		original_image = cv2.imread(self.file_model_png, 1)
		scaled_image = cv2.resize(original_image, (ImageModelConfiguration.width,ImageModelConfiguration.height))  
		
		success, encoded_image = cv2.imencode('.png', scaled_image)
		imgbytes = encoded_image.tobytes()
		self.imageContainer.update(data=imgbytes)

	def displayFinishedModelImage(self):
		original_image = cv2.imread(self.file_model_png, 1)
		scaled_image = cv2.resize(original_image, (ImageModelConfiguration.width,ImageModelConfiguration.height))  
		
		success, encoded_image = cv2.imencode('.png', scaled_image)
		imgbytes = encoded_image.tobytes()
		self.imageContainer.update(data=imgbytes)
		
	def clearModelImageResult(self):
		self.imageContainer.update(data=[])
