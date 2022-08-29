import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QFormLayout, QGridLayout, QMessageBox, QPushButton, QLineEdit, QPlainTextEdit, QSpinBox

### Implementacja "Paint'a"


# Wielkością oraz kolorem pisaka
PEN_WIDTH = 10
PEN_COLOR = Qt.white

# Wielkość pola do rysowania PIXMAP_SIZE x PIXMAP_SIZE
PIXMAP_SIZE = 256

# Prosta implementacja Painta
class Paint(QtWidgets.QMainWindow):

    def __init__(self, predict_function):
        """
        predict_function - funkcja wywoływana po zakończeniu rysowania. 
        Powinna zwracać wartość (liczbę), która została zwrócona przez sieć neuronową
        """
        super().__init__()

        # Główny widżet przechowujący layout
        self.window = QWidget()

        # Tworzenie okna w którym będzie możliwość rysownia
        self.paint = QtWidgets.QLabel()
        self.canvas = QtGui.QPixmap(PIXMAP_SIZE, PIXMAP_SIZE)
        self.paint.setPixmap(self.canvas)

        # Tworzenie layout'u przechowującego:
        # - okno do rysowania
        # - przycisk do wyczyszczenia obrazu
        # - okno wyświetlające odpowiedź sieci neuronowej
        self.layout = QGridLayout()
        self.prediction = QLineEdit()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)   

        self.layout.addWidget(self.prediction,1,0) 
        self.layout.addWidget(self.clear_button,1,1) 
        self.layout.addWidget(self.paint,0,0) 

        self.prediction.setDisabled(1)
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)

        # Zmienne przechowujące ostatnią pozycję myszy 
        self.last_x = None
        self.last_y = None

        self.predict_function = predict_function

    def clear(self):
        """
        Funkcja czyszcząca pole rysowania
        """
        self.paint.pixmap().fill(Qt.black)
        self.update()

    def mouseMoveEvent(self, e):
        """
        Funkcja wywoływana, podczas poruszania wskaźnikiem myszy z wciśniętym przyciskiem myszy.
        """
        if self.last_x is None: 
            self.last_x = e.x()
            self.last_y = e.y()
            return 

        # Obsługa rysowania 
        painter = QPainter(self.paint.pixmap())
        painter.setPen(QPen(PEN_COLOR, PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            
        x = self.paint.geometry().x()
        y = self.paint.geometry().y()

        painter.drawLine(self.last_x - x, self.last_y - y, e.x() - x, e.y() - y)

        self.update()
        

        self.last_x = e.x()
        self.last_y = e.y()
        #
        #
        #
        #
        # Ustawienie w polu tekstowym wartości zwróconej przez sieć neuronową
        # self.paint.pixmap().toImage() - zwraca obiekt QImage
        self.prediction.setText(str(self.predict_function(self.paint.pixmap().toImage())))

    def mouseReleaseEvent(self, e):
        """
        Funkcja wywoływana, podczas puszczenia przycisku myszy.
        """
        self.last_x = None
        self.last_y = None



import Keras_PyQt_Paint_Model as kppm

app = QtWidgets.QApplication(sys.argv)
window = Paint(lambda x: kppm.predict(x, kppm.get_model()))
window.show()
app.exec_()