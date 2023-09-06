import sklearn.utils._weight_vector
import PySimpleGUI as sg
import json 

#for window application

import win32gui, win32con
hide = win32gui.GetForegroundWindow()
win32gui.ShowWindow(hide , win32con.SW_HIDE)

config = ""
with open("cfg/config.txt", "r") as file:
    config = json.loads(file.read())

theme_selected = config["theme-selected"]
if theme_selected == "" :
	theme_selected = "SystemDefault"
sg.theme(theme_selected)

import matplotlib
matplotlib.use("TkAgg")

#import gui.MainLayout as overview
#mainLayout = overview.MainLayout()
from gui.MainLayout import MainLayout 
mainLayout = MainLayout()

#import dp.DataPreparation as dp
#dataPreparation = dp.DataPreparation()
from dp.DataPreparation import DataPreparation
dataPreparation = DataPreparation()

dataset = []
selectedSurveyFields = []
headerSurveyFields = []

selectedVisFields = []
headerVisFields = []

#train model
selectedTrainModelFields = []
headerTrainModelFields = []
Model = None

#prediction result
str_prediction_result = ""

window = sg.Window(config["Title"],mainLayout.getLayout(),icon=config["favicon"], return_keyboard_events=True) 

def getMultiLineString(header):
	if len(header)>0:
		order = []
		for item in header:
			for i in range(0,len(dataset[0])):
				if item == dataset[0][i]:
					order.append(i)

	headerString = ""
	for i in range(0,len(order)):
		headerString = headerString + dataset[0][order[i]] + "\t\t"
	headerString = headerString + "\n"

	dataString = ""
	for row in dataset[1]:
		for i in range(0,len(order)):
			dataString = dataString + str(row[order[i]]) + "\t\t"
		dataString = dataString + "\n"
	
	data = headerString + dataString;
	return data

def statusBarUpdateMessage(message,color):
	window["-StatusLog-"].update(message,text_color=color)

while True:
	evt, values = window.read()
	#print(evt,values)

	if evt == sg.WIN_CLOSED:
		break

	#the part of surveying data
	if evt == "-evtOnLoadCsvFile-":
		try :
			if values["-pathToCsvDataSource-"] != "" :
			
				csvFileName = values["-pathToCsvDataSource-"];
				
				dataset = dataPreparation.readDataSource(csvFileName)
				selectedSurveyFields = []
				headerSurveyFields = dataset[0].copy()
				
				selectedVisFields = []
				headerVisFields = dataset[0].copy()
				
				selectedTrainModelFields = []
				headerTrainModelFields = dataset[0].copy()
				
				window["-listSelectingDataSetHeader-"].update(dataset[0])
				window["-listSelectedDataSetHeader-"].update([])

				window["-listVisSelectingDataSetHeader-"].update(dataset[0])
				window["-listVisSelectedDataSetHeader-"].update([])

				window["-listTrainSelectingDataSetHeader-"].update(dataset[0])
				window["-listTrainSelectedDataSetHeader-"].update([])
				
				#data = getMultiLineString(dataset[0])
				#window["-listTbDataSet-"].update(data)
				
				window["-listTbDataSet-"].update(dataset[2])
				
				statusBarUpdateMessage("Load data success","white")

		except Exception as e:
			statusBarUpdateMessage(str(e),"red")
				
	if evt == "-evtSelectingFields-":
		try :
			if len(values["-listSelectingDataSetHeader-"][0]) > 0 :
			
				selectedSurveyFields.append(values["-listSelectingDataSetHeader-"][0])
				window["-listSelectedDataSetHeader-"].update(selectedSurveyFields)
				
				headerSurveyFields.remove(values["-listSelectingDataSetHeader-"][0])
				window["-listSelectingDataSetHeader-"].update(headerSurveyFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")	
				
	if evt == "-evtUnSelectFields-":
		try :
			if len(values["-listSelectedDataSetHeader-"][0]) > 0 :

				selectedSurveyFields.remove(values["-listSelectedDataSetHeader-"][0])
				window["-listSelectedDataSetHeader-"].update(selectedSurveyFields)
				
				headerSurveyFields.append(values["-listSelectedDataSetHeader-"][0])
				window["-listSelectingDataSetHeader-"].update(headerSurveyFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")
				
	if evt == "-evtDisplaySelectedDataSet-":
		try :
			if len(selectedSurveyFields)>0:
				data = getMultiLineString(selectedSurveyFields)
				window["-listTbDataSet-"].update(data)
				
				statusBarUpdateMessage("Display selected data success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")

	if evt == "-evtCleanSelectedDataSet-":
		try:
			dataPreparation.cleanData(selectedSurveyFields)
			data = getMultiLineString(selectedSurveyFields)
			window["-listTbDataSet-"].update(data)
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")
			
	#the part of visualizing data
	if evt == "-evtVisSelectingFields-":
		try :
			if len(values["-listVisSelectingDataSetHeader-"][0]) > 0 :
			
				selectedVisFields.append(values["-listVisSelectingDataSetHeader-"][0])
				window["-listVisSelectedDataSetHeader-"].update(selectedVisFields)
				
				headerVisFields.remove(values["-listVisSelectingDataSetHeader-"][0])
				window["-listVisSelectingDataSetHeader-"].update(headerVisFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")	
				
	if evt == "-evtVisUnSelectFields-":
		try :
			if len(values["-listVisSelectedDataSetHeader-"][0]) > 0 :

				selectedVisFields.remove(values["-listVisSelectedDataSetHeader-"][0])
				window["-listVisSelectedDataSetHeader-"].update(selectedVisFields)
				
				headerVisFields.append(values["-listVisSelectedDataSetHeader-"][0])
				window["-listVisSelectingDataSetHeader-"].update(headerVisFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")	

	if evt == "-evtDisplayVisSelectedDataSet-":		
		try :
			if len(selectedVisFields)>0:
				if values["-cmdVisChart-"] == "Histogram":
					dataPreparation.displayHist(window["-imageVisDataSource-"],selectedVisFields)
					statusBarUpdateMessage("Display chart success","white")

				elif values["-cmdVisChart-"] == "Scatter":
					dataPreparation.displayScatter(window["-imageVisDataSource-"],selectedVisFields)
					statusBarUpdateMessage("Display chart success","white")
				elif values["-cmdVisChart-"] == "ParallelCoords":	
					dataPreparation.displayParallelCoords(window["-imageVisDataSource-"],selectedVisFields)
					statusBarUpdateMessage("Display chart success","white")
		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")
			
	#the part of Train model
	
	if evt == "-evtTrainSelectingFields-":
		try :
			if len(values["-listTrainSelectingDataSetHeader-"][0]) > 0 :

				selectedTrainModelFields.append(values["-listTrainSelectingDataSetHeader-"][0])
				window["-listTrainSelectedDataSetHeader-"].update(selectedTrainModelFields)
				
				headerTrainModelFields.remove(values["-listTrainSelectingDataSetHeader-"][0])
				window["-listTrainSelectingDataSetHeader-"].update(headerTrainModelFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")		
			
	if evt == "-evtTrainUnSelectFields-":
		try :
			if len(values["-listTrainSelectedDataSetHeader-"][0]) > 0 :

				selectedTrainModelFields.remove(values["-listTrainSelectedDataSetHeader-"][0])
				window["-listTrainSelectedDataSetHeader-"].update(selectedTrainModelFields)
				
				headerTrainModelFields.append(values["-listTrainSelectedDataSetHeader-"][0])
				window["-listTrainSelectingDataSetHeader-"].update(headerTrainModelFields)
				
				statusBarUpdateMessage("Select field success","white")
		except Exception as e:
			statusBarUpdateMessage(str(e),"red")		
	
	if values["-cmdApproaches-"]:
	#if evt == "-evtSelectApproaches-":
	
		if values["-cmdApproaches-"] == "Classification":
			window["-cmdAlgorithm-"].update(values = ["DecisionTree","NeuralNetwork","SVM","NaiveBayes","KNeighborsClassifier"])
		elif values["-cmdApproaches-"] == "Regression":
			window["-cmdAlgorithm-"].update(values = ["LinearRegression","PolynomialRegression"])
		elif values["-cmdApproaches-"] == "Clustering":
			window["-cmdAlgorithm-"].update(values = ["KMeans","MeanShift"])	
	
	if evt == "-evtTuneParameters-":
	#if values["-cmdAlgorithm-"] : 
		try:
			#clear model image 
			if Model:
				Model.clearModelImageResult()
			
			algorithm = values["-cmdAlgorithm-"]
			
			if values["-cmdAlgorithm-"] == "LinearRegression" :
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				from tm.Regression.LinearRegressionModel import LinearRegressionModel as LinearRegressionModel
				Model = LinearRegressionModel(dataPreparation.data[:,column[0]],
					dataPreparation.data[:,column[1]],
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())
				
			elif values["-cmdAlgorithm-"] == "PolynomialRegression" :
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				from tm.Regression.PolynomialRegressionModel import PolynomialRegressionModel as PolynomialRegressionModel
				Model = PolynomialRegressionModel(dataPreparation.data[:,column[0]],
					dataPreparation.data[:,column[1]],
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())
				
			elif values["-cmdAlgorithm-"] == "DecisionTree" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				trainingYSet = dataPreparation.data[:,column[-1]]
				
				for i in range(1,len(column)-1):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
				
				from tm.Classification.DecisionTreeModel import DecisionTreeModel as DecisionTreeModel
				Model = DecisionTreeModel(trainingXSet,
					trainingYSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())

			elif values["-cmdAlgorithm-"] == "NeuralNetwork" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				trainingYSet = dataPreparation.data[:,column[-1]]
				
				for i in range(1,len(column)-1):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
				
				from tm.Classification.NeuralNetworkModel import NeuralNetworkModel as NeuralNetworkModel
				Model = NeuralNetworkModel(trainingXSet,
					trainingYSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())

			elif values["-cmdAlgorithm-"] == "SVM" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				trainingYSet = dataPreparation.data[:,column[-1]]
				
				for i in range(1,len(column)-1):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
				
				from tm.Classification.SvmModel import SvmModel as SvmModel
				Model = SvmModel(trainingXSet,
					trainingYSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())		
				
			elif values["-cmdAlgorithm-"] == "NaiveBayes" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				trainingYSet = dataPreparation.data[:,column[-1]]
				
				for i in range(1,len(column)-1):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
				
				from tm.Classification.NaiveBayesModel import NaiveBayesModel as NaiveBayesModel
				Model = NaiveBayesModel(trainingXSet,
					trainingYSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())		

			elif values["-cmdAlgorithm-"] == "KNeighborsClassifier" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				trainingYSet = dataPreparation.data[:,column[-1]]
				
				for i in range(1,len(column)-1):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
				
				from tm.Classification.KNeighborsModel import KNeighborsModel as KNeighborsModel
				Model = KNeighborsModel(trainingXSet,
					trainingYSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())		
				
			elif values["-cmdAlgorithm-"] == "KMeans" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				
				for i in range(1,len(column)):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
					
				from tm.Clustering.KMeansClusteringModel import KMeansClusteringModel as KMeansClusteringModel
				Model = KMeansClusteringModel(trainingXSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())

			elif values["-cmdAlgorithm-"] == "MeanShift" :
				import numpy as np
				column = dataPreparation.getSelectedColumnIndexs(selectedTrainModelFields)
				trainingXSet = np.array([dataPreparation.data[:,column[0]]])
				
				for i in range(1,len(column)):
					a = np.array([dataPreparation.data[:,column[i]]])
					trainingXSet = np.concatenate((trainingXSet, a))
					
				from tm.Clustering.MeanShiftClusteringModel import MeanShiftClusteringModel as MeanShiftClusteringModel
				Model = MeanShiftClusteringModel(trainingXSet,
					selectedTrainModelFields,
					window["-modelImageResult-"])
				window["-txtConfiguration-"].update(Model.getConfiguration())
				
			window["-cmdAlgorithm-"].update(algorithm)
			
		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")

	if evt == "-evtTrainModel-":
		try:
			algorithm = values["-cmdAlgorithm-"]				
			#train model using configuration			
			cfg_json_str = values["-txtConfiguration-"]
			result = Model.fit(cfg_json_str)
			window["-txtModelResult-"].update(result)
						
			#draw image model
			if Model.displayChart > 0:
				Model.drawFitModel()
			
			statusBarUpdateMessage("Train model success","white")
			window["-cmdAlgorithm-"].update(algorithm)

		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")

	if evt == "-evtSaveModel-":
		try:
			import pickle
			modelFileName = values["-pathToSaveModel-"]
			pickle.dump(Model.model, open(modelFileName, "wb"))
			statusBarUpdateMessage("Save model success","white")
			Model.savePredictExample(modelFileName)
		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")
	
	# To predict using model	
	if evt == "-isPolynomial-":
		if values["-isPolynomial-"] == True :
			window["-cmdPolyDegree-"].update(values=[2,3,4,5,6,7,8,9,10])
		else:
			window["-cmdPolyDegree-"].update(values=[])
		
	if evt == "-evtOnPredictData-":
		try:
			import pickle
			import numpy as np
			import pandas as pd
			#load model file
			modelFileName = values["-pathToModelSource-"]
			loaded_model = pickle.load(open(modelFileName, 'rb'))		
			#print(loaded_model)

			#csv file path for prediction
			str_prediction_result = ""
			csvFile = values["-pathToCsvPredictSource-"]

			#load header for generate result
			str_header = ""
			df_header = pd.read_csv(csvFile, nrows=1)
			for column in df_header.columns:
				str_header = str_header + str(column) + ","
			str_header = str_header + "result\n"
			#print(str_header)

			#load data for predicting
			df = pd.read_csv(csvFile, sep=',', engine='python')
			questions = df[:].values.tolist()
			questions = np.array(questions)
			#print(questions)
			
			#predict 
			result = []
			if values["-isPolynomial-"] != True :
				result = loaded_model.predict(questions)
			else:
				if values["-cmdPolyDegree-"] > 1:
					deg = values["-cmdPolyDegree-"]
					from sklearn.preprocessing import PolynomialFeatures
					poly_reg = PolynomialFeatures(degree = deg)
					questions = poly_reg.fit_transform(questions)
					result = loaded_model.predict(questions)

			str_results = ""
			for i in range(len(questions)):
				#str_results = str_results + str(np.array2string(questions[i])) +" => "+str(result[i])+ "\n"
				params = questions[i]
				for param in params:
					str_results = str_results + str(param) + ","
				str_results = str_results + str(result[i]) + "\n"

			str_prediction_result = str_header + str_results
			#print(str_prediction_result)

			window["-listPredictResult-"].update(str_prediction_result)
		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")

	if evt == "-evtExportPredictionResult-":
		try:
			csvFile = values["-pathToSavePredictionResult-"]
			f = open(csvFile, "w")
			f.write(str_prediction_result)
			f.close()
			statusBarUpdateMessage("Export prediction result in csv format success.","white")
		except Exception as e:
			sg.Popup(str(e))
			statusBarUpdateMessage(str(e),"red")
		
window.close()











