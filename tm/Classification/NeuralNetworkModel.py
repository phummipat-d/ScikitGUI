import json 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from tm.Classification.ClassificationModel import ClassificationModel

class NeuralNetworkModel(ClassificationModel):

	file_model_png = "tmp/finished.png"
	hidden_layer_sizes=100
	activation='relu'
	solver='adam'
	alpha=0.0001
	batch_size='auto'
	learning_rate='constant'
	learning_rate_init=0.001
	power_t=0.5
	max_iter=200
	
	def __init__(self,x,y,selectedTrainModelFields,imageContainer):
		super().__init__(x,y,selectedTrainModelFields,imageContainer)
		self.model = MLPClassifier()
		
	def fit(self,cfg_json_string):
		#to split data for training and testing 
		self.setConfiguration(cfg_json_string)

		if self.displayChart > 0:
			self.preLoadImage()
		else:
			self.clearModelImageResult()
			
		#to split data for training and testing 
		test_size = self.amountOfTestingDataSet/100.0
		if test_size < 1.0:
			self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size)		
		else:
			self.x_train = self.x
			self.y_train = self.y	
			self.x_test = self.x
			self.y_test = self.y
		
		#the part for fitting model
		self.model.fit(self.x_train,self.y_train)
		self.y_pred = self.model.predict(self.x_test)

		str_result =  "Accuracy:"+str(metrics.accuracy_score(self.y_test, self.y_pred))+"\n\n"
		str_result =  str_result + "Confusion Matrix\n"
		conf_matix = confusion_matrix(self.y_test, self.y_pred)	
		str_result =  str_result + str(conf_matix)
		
		return str_result
		
	def getConfiguration(self):
		cfg_json_str = '{"amountOfTrainingDataSet":"'+str(self.amountOfTrainingDataSet)+'","amountOfTestingDataSet":"'+str(self.amountOfTestingDataSet)+'","displayChart":"'+str(self.displayChart)+'","hidden_layer_sizes":"'+str(self.hidden_layer_sizes)+'","activation":"'+str(self.activation)+'","solver":"'+str(self.solver)+'","alpha":"'+str(self.alpha)+'","batch_size":"'+str(self.batch_size)+'","learning_rate":"'+str(self.learning_rate)+'","learning_rate_init":"'+str(self.learning_rate_init)+'","max_iter":"'+str(self.max_iter)+'","power_t":"'+str(self.power_t)+'"}'
		return cfg_json_str
	
	def setConfiguration(self,cfg_json_str):
		cfg_json = json.loads(cfg_json_str)
		self.amountOfTrainingDataSet = int(cfg_json["amountOfTrainingDataSet"])
		self.amountOfTestingDataSet = int(cfg_json["amountOfTestingDataSet"])
		self.displayChart = int(cfg_json["displayChart"])
		self.hidden_layer_sizes = int(cfg_json["hidden_layer_sizes"])
		self.activation = cfg_json["activation"]
		self.solver = cfg_json["solver"]
		self.alpha = float(cfg_json["alpha"])
		self.batch_size = cfg_json["batch_size"]
		self.learning_rate = cfg_json["learning_rate"]
		self.learning_rate_init = float(cfg_json["learning_rate_init"])
		self.power_t = float(cfg_json["power_t"])
		self.max_iter = int(cfg_json["max_iter"])
	
	def savePredictExample(self,modelFileName):
		example_header = "This model is used to predict "+self.selectedTrainModelFields[-1]+" using "+str(self.selectedTrainModelFields[0:-1])+"\n"
		example_header = example_header + "You can prepare your data using the format below and save them in csv format.\n"
		example_header = example_header + "=====================================\n\n"
		example_string = ""
		
		#print(example_header)
	
		for i in range(0,len(self.selectedTrainModelFields)-1):
			example_string = example_string + self.selectedTrainModelFields[i]+","
		example_string = example_string.strip(',') + "\n"
		#print(example_string)
		
		for i in range(0,len(self.selectedTrainModelFields)-1):
			example_string = example_string + str(self.x_test[0,i])+","
		example_string = example_string.strip(',') + "\n"
		#print(example_string)
		
		outputReadme = example_header+ example_string + "\n\n"
		
		f = open(modelFileName+"_Readme.txt", "w")
		f.write(outputReadme)
		f.close()
		
		outputExampleData = example_string	
		f = open(modelFileName+"_example.csv", "w")
		f.write(outputExampleData)
		f.close()
	
	def drawFitModel(self):
		self.displayFinishedModelImage()