import PySimpleGUI as sg

class DataPreparationLayout:

	layout = [[
		sg.Frame("Select Data Source",[[
			sg.Column([
				[sg.Text("Select csv File")]
			]),
			sg.Column([
				[sg.In(key='-pathToCsvDataSource-')]
			]),				
			sg.Column([
				[sg.FileBrowse("Select",target='-pathToCsvDataSource-',file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))]
			]),
			sg.Column([
				[sg.Button(" Open ",key ="-evtOnLoadCsvFile-")]
			])
		]])
	],[
		sg.Frame("Data Source",[[
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listSelectingDataSetHeader-")]
			]),
			sg.Column([
				[sg.Button(" >> ",key ="-evtSelectingFields-")],
				[sg.Button(" << ",key ="-evtUnSelectFields-")]
			]),				
			sg.Column([
				[sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-listSelectedDataSetHeader-")]
			])
		],
			[
				sg.Button("Display Selected Fields",key ="-evtDisplaySelectedDataSet-"),
				sg.Button("Clean Selected Fields",key ="-evtCleanSelectedDataSet-")
			],
			[sg.Multiline(size=(83, 20), font='courier 10', background_color='white', text_color='black', key='-listTbDataSet-')]
		])
	]]

	def getLayout(self):
		return self.layout