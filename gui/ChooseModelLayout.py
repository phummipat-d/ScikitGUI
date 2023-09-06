import PySimpleGUI as sg

class ChooseModelLayout:
	layout = [[
		sg.Frame('',[
		[
			sg.Text("ChooseModelLayout")
		]
		])
	]]
	
	def getLayout(self):
		return self.layout