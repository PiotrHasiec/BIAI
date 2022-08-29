from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget
from PyQt5.QtWidgets import QLabel, QFileDialog, QPushButton, QLineEdit, QPlainTextEdit, QSpinBox,QVBoxLayout
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap
import PyQt5.QtWidgets
import numpy as np
import Keras_Model_functions as ppm
from PyQt5.QtCore import Qt
import cv2
# Tworzenie klasy głównego okna aplikacji dziedziczącej po QMainWindow

class Window(QMainWindow):
    #Dodanie konstruktora przyjmującego okno nadrzędne
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filename = ""
        self.setWindowTitle('BIAI Face Generator')
        self.resize(400,400)
        self.createMenu()
        self.createTabs()
        
    
    # Funkcja dodająca pasek menu do okna
    def createMenu(self):
        # Stworzenie paska menu
        self.menu = self.menuBar()
        # Dodanie do paska listy rozwijalnej o nazwie File
        self.FileMenu = self.menu.addMenu("File")
        self.Z1Menu = self.menu.addMenu("Z1")

        
        self.FileMenu.addAction('Exit', self.close)
        self.Z1Menu.addAction('Wybierz plik', self.wyb_obr)


    
    def wyb_obr(self):
      fileName, selectedFilter = QFileDialog.getOpenFileName(self, "Wybierz plik obrazu",  "Początkowa nazwa pliku", "All Files (*);;Python Files (*.py);; PNG (*.png)")
      if fileName:
         
         self.pixmap = QPixmap(fileName)
         self.label1.setPixmap( self. pixmap)
         self.label1.resize(self.pixmap.width(),  self.pixmap.height())
         self.resize(  self.pixmap.width() +10,  self.pixmap.height()+80)
         
         self.show()
       
    
    
    # Funkcja dodająca wenętrzeny widżet do okna
    def createTabs(self):
        # Tworzenie widżetu posiadającego zakładki
        self.tabs = QTabWidget()
        
        # Stworzenie osobnych widżetów dla zakładek
        self.tab_1 = QWidget()
        self.tab_2 = QWidget()
        self.tab_3 = QWidget()
        
        # Dodanie zakładek do widżetu obsługującego zakładki
        # Zakładka 1
        self.tabs.addTab(self.tab_1, "Pierwsza zakładka") 
        self.button = QPushButton("Generuj")
        self.button.setMaximumHeight(500)
        self.button.clicked.connect(self.predict_rd)
        self.button2 = QPushButton("Generuj uwzględniając korelacje")
        self.button2.setMaximumHeight(500)
        self.button2.clicked.connect(self.predict_rd2)
        layoutz1 =  QVBoxLayout()
        self.pixmap = QPixmap(300,300)
        layoutz1.addWidget(self.button)
        layoutz1.addWidget(self.button2)
        self.tab_1.setLayout(layoutz1)
        # Zakładka 2
        self.model,self.dec =ppm.get_model("model14")
        
        self.means, self.stds, self.c  = ppm.get_values(self.model)
        
        layoutz2 =  QVBoxLayout()
        self.tabs.addTab(self.tab_2, "Druga zakładka")
        self.sliders = []
        for i in range(5*32):
            
            slider = PyQt5.QtWidgets.QSlider(Qt.Horizontal)
            
            self.sliders.append(slider)
            if self.stds[i]>0 :
                slider.valueChanged.connect(self.predict)
                slider.setFocusPolicy(Qt.StrongFocus)
                slider.setMinimum( int(10000*(self.means[i]-3*self.stds[i])))
                slider.setMaximum(int(10000*(self.means[i]+3*self.stds[i])))
                layoutz2.addWidget(slider)
             
        
        
        
       
        self.tab_2.setLayout(layoutz2)
        
        self.setCentralWidget(self.tabs)

    def predict(self):
        input = [i.value()/10000 for i in self.sliders]
        
        self.pixmap =ppm.predict(input,self.dec)
        cv2.imshow("Face",self.pixmap)
    def predict_rd(self):

        input = [np.random.standard_normal() for i in range(len(self.sliders))]
        #input = np.random.uniform(0,1.00,len(self.sliders))
        x = np.dot(self.c, input) +self.means

        
        self.pixmap =ppm.predict(x,self.dec)
        cv2.imshow("Face",self.pixmap)

    def predict_rd2(self):

        input = [np.random.normal(self.means[i],self.stds[i]) for i in range(len(self.sliders))]
          
        
        
        self.pixmap =ppm.predict(input,self.dec)
        cv2.imshow("Face",self.pixmap)
    def open(self):
        fileName, selectedFilter = QFileDialog.getOpenFileName(self, "Wybierz plik obrazu",  "Początkowa nazwa pliku", "All Files (*);;Python Files (*.py);; TXT (*.txt)")
        if fileName:
            self.text_Z2.clear()
            text=open(fileName).read()
            self.text_Z2.setPlainText(text)

            
   
    
    


        
            
    
        
# Uruchomienie okna
app = QApplication([])
win = Window()
win.show()
app.exec_()

