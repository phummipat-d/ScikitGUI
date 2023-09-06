import PySimpleGUI as sg

class DataVisualizationLayout:
	layout = [[
		sg.Frame("Data Set",[[
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listVisSelectingDataSetHeader-")]
			]),
			sg.Column([
				[sg.Button(" >> ",key ="-evtVisSelectingFields-")],
				[sg.Button(" << ",key ="-evtVisUnSelectFields-")]
			]),				
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listVisSelectedDataSetHeader-")]
			])
		],
			[
				sg.Text("Select"),
				sg.Combo(values=["Histogram","Scatter","ParallelCoords"], key='-cmdVisChart-'),
				sg.Button("Display Chart",key ="-evtDisplayVisSelectedDataSet-")
			],
			[sg.Image(key="-imageVisDataSource-")]
		])
	]]

	def getLayout(self):
		return self.layout