import sklearn.utils._weight_vector
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer 

import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
import io

class DataPreparation:
	figure_width = 670
	figure_height = 400
	data = []
	header = []	
	
	figure_canvas_agg = None
	
	def readDataSource(self,filename):
		df = pd.read_csv(filename, sep=',', engine='python')
		
		#convert df.info() to string
		buf = io.StringIO()
		df.info(buf=buf)
		data_info = buf.getvalue()
		
		#read header and data
		self.header = list(df.columns)
		#self.data = df[1:].values.tolist()
		self.data = df[:].values.tolist()
		self.data = np.array(self.data)
		#print(self.data)
		
		return [self.header,self.data,data_info]
	
	def cleanData(self,selectedColumnIndexes) :
		columns = self.getSelectedColumnIndexs(selectedColumnIndexes)
		for i in columns:
			imputer = SimpleImputer(missing_values=np.nan,strategy = "mean") 
			dataCleaned = self.data[:,i]
			dataCleaned = dataCleaned.astype(np.float)
			imputer.fit([dataCleaned])
			dataCleaned = imputer.transform([dataCleaned])
			dataCleaned = dataCleaned[0]
			
			for row in range(0,len(dataCleaned)):
				self.data[row,i] = dataCleaned[row]
		
	def displayScatter(self,imageContainer,selectedColumnIndexes):
		columns = self.getSelectedColumnIndexs(selectedColumnIndexes)
		x_np = self.data[:,columns[0]]
		x = x_np.astype(np.float)
		y_np = self.data[:,columns[1]]
		y = y_np.astype(np.float)
		
		figure, ax = plt.subplots()
		ax.scatter(x, y, alpha=0.3)
		
		xmax = np.amax(x)
		xmin = np.amin(x)
		ymax = np.amax(y)
		ymin = np.amin(y)
		
		ax.axes.xaxis.set_ticks([xmin,(xmax+xmin)/2,xmax])
		ax.axes.yaxis.set_ticks([ymin,(ymax+ymin)/2,ymax])

		plt.xlabel(self.header[columns[0]])
		plt.ylabel(self.header[columns[1]])
		
		filename = "tmp/TmpScatter.png"
		plt.savefig(filename,
			format='png',
			bbox_inches = "tight",
			pad_inches=0.0,
			dpi = 300)
		
		self.displayVisImage(imageContainer,filename)
				
	def displayHist(self,imageContainer,selectedColumnIndexes):
		columns = self.getSelectedColumnIndexs(selectedColumnIndexes)
		
		x = self.data[:,columns[0]]
		figure, ax = plt.subplots()

		ax.axes.xaxis.set_ticks([])
		
		plt.xlabel('Value')
		plt.ylabel('Frequency')

		plt.hist(x)

		filename = "tmp/TmpHistogram.png"
		plt.savefig(filename,
			format='png',
			bbox_inches = "tight",
			pad_inches=0.0,
			dpi = 300)
		
		self.displayVisImage(imageContainer,filename)
		
	def displayParallelCoords(self,imageContainer,selectedColumnIndexes):
	
		tempParallelleCoordsFile = "tmp/TmpParallelCoords.png"
		#print(selectedColumnIndexes)
		column = self.getSelectedColumnIndexs(selectedColumnIndexes)
		seletedData = np.array([self.data[:,column[0]]])
		for i in range(1,len(column)):
			a = np.array([self.data[:,column[i]]])
			seletedData = np.concatenate((seletedData, a))		
		seletedData = seletedData.transpose().astype(np.float)
		
		df = pd.DataFrame(seletedData,columns=selectedColumnIndexes)
		figure = px.parallel_coordinates(df,color = df.columns[0], dimensions = list(df.columns),
			color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
		figure.write_image(tempParallelleCoordsFile)
		
		self.displayVisImage(imageContainer,tempParallelleCoordsFile)
		
	def displayVisImage(self,imageContainer,filename):
		original_image = cv2.imread(filename, 1)
		scaled_image = cv2.resize(original_image, (self.figure_width,self.figure_height))  
		
		success, encoded_image = cv2.imencode('.png', scaled_image)
		imgbytes = encoded_image.tobytes()
		imageContainer.update(data=imgbytes)		
		
	def getSelectedColumnIndexs(self,selectedColumns):
		if len(selectedColumns)>0:
			columns = []
			for item in selectedColumns:
				for i in range(0,len(self.header)):
					if item == self.header[i]:
						columns.append(i)
		return columns
		
		
		
		
		
		
		
		
		