# ScikitGUI

portable version can be downloaded here [https://drive.google.com/file/d/1MSQJzPIHZmlEtqeS2rezPjdBM9mdAoq9/view?usp=drive_link](https://drive.google.com/file/d/1MSQJzPIHZmlEtqeS2rezPjdBM9mdAoq9/view?usp=sharing)

<b>installation steps:</b><br>
1.create enveronement for ScikitGUI using python 3.6.13 for example : conda create -n ScikitGUI python=3.6.13 anaconda<br>
2.install package belowing:<br>
  conda install scikit-learn==0.24.1<br>
  conda install matplotlib==3.3.4<br>
  conda install pandas==1.1.5<br>
  conda install plotly==5.6.0<br>
  pip install pysimplegui==4.38.0<br>
  pip install opencv-python==4.5.2.52<br>
  pip install kaleido<br>
  pip install pyinstaller<br>
  pip install pywin32<br>

<b>building exe steps:</b><br>
1.activate ScikitGUI<br>
2.pyinstaller --onefile ScikitGUI.py<br>
3.ScikitGUI.exe will show in folder <b>dist</b>
