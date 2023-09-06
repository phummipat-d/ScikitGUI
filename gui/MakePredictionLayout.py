import PySimpleGUI as sg

class MakePredictionLayout:
	layout = [[
		sg.Frame("",[
			[
				sg.Column([
					[sg.Text("Select model   ")]
				]),
				sg.Column([
					[sg.In(key='-pathToModelSource-')]
				]),
				sg.Column([
					[sg.FileBrowse("Select",target='-pathToModelSource-',file_types=(("Scikit Learn", "*.sav"),))]
				])
			],
			[
				sg.Column([
					[sg.Text("Select csv data")]
				]),
				sg.Column([
					[sg.In(key='-pathToCsvPredictSource-')]
				]),				
				sg.Column([
					[sg.FileBrowse("Select",target='-pathToCsvPredictSource-',file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))]
				])
			],
			[
				sg.Checkbox('Polynomial', key='-isPolynomial-', enable_events=True),
				sg.Text(","),
				sg.Text("Degree"),
				sg.Combo(values=[], key='-cmdPolyDegree-'),
				sg.Text(","),
				sg.Button(" Predict ",key ="-evtOnPredictData-")
			]
		])
	],
		[
			sg.Text("Result")
		],
		[
			sg.Multiline(size=(83, 21), font='courier 10', background_color='white', text_color='black', key='-listPredictResult-')
		],
		[
			sg.In(size=(25,1),key='-pathToSavePredictionResult-'),
			sg.SaveAs("Select",target='-pathToSavePredictionResult-',file_types=(("CSV", "*.csv"),)), 
			sg.Button("Export Prediction",key ="-evtExportPredictionResult-")
		]
	]
	
	def getLayout(self):
		return self.layout