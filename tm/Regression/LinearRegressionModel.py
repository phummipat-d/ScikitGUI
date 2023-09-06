import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tm.Regression.RegressionModel import RegressionModel
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import cv2

class LinearRegressionModel(RegressionModel):
	file_model_png = "tmp/LinearRegressionModel.png"
		
	def __init__(self,x,y,selectedTrainModelFields,imageContainer):
		super().__init__(x,y,selectedTrainModelFields,imageContainer)
		self.model = LinearRegression()
		
	def fit(self,cfg_json_string):
		#to split data for training and testing 
		self.setConfiguration(cfg_json_string)
		
		#print(self.displayChart)
		if self.displayChart > 0:
			self.preLoadImage()
		else:
			self.clearModelImageResult()		
		
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
		
		# The coefficients
		str_result = 'Coefficients:'+str(self.model.coef_)+'\n'
		# The mean squared error
		str_result = str_result + 'Mean squared error: ' + str(mean_squared_error(self.y_test, self.y_pred))+'\n'
		# The coefficient of determination: 1 is perfect prediction
		str_result = str_result + 'Coefficient of determination: '+ str(r2_score(self.y_test, self.y_pred))+'\n'
		
		return str_result
		
	def savePredictExample(self,modelFileName):
		example_header = "This model is used to predict "+self.selectedTrainModelFields[1]+" using "+self.selectedTrainModelFields[0]+"\n"
		example_header = example_header + "You can prepare your data using the format below and save them in csv format.\n"
		example_header = example_header + "=====================================\n\n"
		example_string = ""
		example_string = example_string + self.selectedTrainModelFields[0]+"\n"
		example_string = example_string + str(self.x_test[0,0])+"\n"
		example_string = example_string + str(self.x_test[1,0])+"\n"
		
		outputReadme = example_header+ example_string	
		f = open(modelFileName+"_Readme.txt", "w")
		f.write(outputReadme)
		f.close()
		
		outputExampleData = example_string	
		f = open(modelFileName+"_example.csv", "w")
		f.write(outputExampleData)
		f.close()
	
	def drawFitModel(self):

		figure, ax = plt.subplots()	
		
		ax.scatter(self.x, self.y,  color='blue', alpha=0.3)
		ax.plot(self.x_test, self.y_pred, color='red', linewidth=1.5)

		ax.axes.xaxis.set_ticks([])
		ax.axes.yaxis.set_ticks([])
			
		plt.xlabel(self.selectedTrainModelFields[0])
		plt.ylabel(self.selectedTrainModelFields[1])
		
		ax.axis('tight')
		
		plt.savefig(self.file_model_png,
			format='png',
			bbox_inches = "tight",
			pad_inches=0.0,
			dpi = 300)
		
		self.displayModelChart()
