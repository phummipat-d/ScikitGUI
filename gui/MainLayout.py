import PySimpleGUI as sg
import gui.DataPreparationLayout as dp
import gui.DataVisualizationLayout as vs
import gui.TrainModelLayout as tm
import gui.MakePredictionLayout as mp

dpLayout = dp.DataPreparationLayout()
vsLayout = vs.DataVisualizationLayout()
tmLayout = tm.TrainModelLayout()
mpLayout = mp.MakePredictionLayout()

class MainLayout:
	layout = [
		[
			sg.TabGroup([[
				sg.Tab("Data Preparation",dpLayout.getLayout(),key="-tabDataPreparationLayout-"), 
				sg.Tab("Data Visualization",vsLayout.getLayout(),key="-tabDataVisualizationLayout-"), 
				sg.Tab("Train the Model",tmLayout.getLayout(),key="-tabTrainModelLayout-"),
				sg.Tab("Make Predictions",mpLayout.getLayout(),key="-tabMakePredictionLayout-")
			]])
		],
		[
			sg.Text("Status :"),
			sg.Text(" "*150,key="-StatusLog-")
		]
	]
		
	def getLayout(self):
		return self.layout