import PySimpleGUI as sg

class TrainModelLayout:
	layout = [[
		sg.Frame('Select Data',[[
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listTrainSelectingDataSetHeader-")]
			]),
			sg.Column([
				[sg.Button(" >> ",key ="-evtTrainSelectingFields-")],
				[sg.Button(" << ",key ="-evtTrainUnSelectFields-")]
			]),				
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listTrainSelectedDataSetHeader-")]
			])
		]])
	],[
		sg.Frame('Select Model',[[
			sg.Text("Approaches"),
			sg.Combo(values=["Classification","Regression","Clustering"], key='-cmdApproaches-',enable_events=True),
			#sg.Button(" Select ",key ="-evtSelectApproaches-"),
			sg.Text("Algorithm"),
			sg.Combo(values=[], key='-cmdAlgorithm-'),
			sg.Button("Tune parameters",key ="-evtTuneParameters-"),
			sg.Button(" Train Model ",key ="-evtTrainModel-")
		]])
	],[
		sg.Frame('Configuration',[[
			sg.Multiline(size=(40,10), font='courier 10', background_color='white', text_color='black', key='-txtConfiguration-'),
			sg.VSeperator(),
			sg.Image(size=(320,150),key="-modelImageResult-")
			#sg.Canvas(size=(320,150),key="-canvasModelResult-")
		]])
	],[
		sg.Frame('Result',[[
			sg.Multiline(size=(83,5), font='courier 10', background_color='white', text_color='black', key='-txtModelResult-')
		],[
			sg.In(size=(25,1),key='-pathToSaveModel-'),
			sg.SaveAs("Select",target='-pathToSaveModel-',file_types=(("Scikit Learn", "*.sav"),)), 
			sg.Button("Save Model",key ="-evtSaveModel-")
		]])	
	]]
	
	def getLayout(self):
		return self.layout