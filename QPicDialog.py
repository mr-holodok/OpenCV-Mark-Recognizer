from PyQt5.QtWidgets import (QDialog, QLabel, QPushButton, QVBoxLayout)
from PyQt5.QtGui import (QFont, QPixmap)
from PyQt5.QtCore import Qt


class QPicDialog(QDialog):
    """class for viewing results"""

    def __init__(self, parent, img_path):
        super().__init__(parent)
        self.img_path = img_path
        self.initUI()

    def initUI(self):
        font = QFont()
        font.setPointSize(12)

        OKBtn = QPushButton(" OK ", self)
        OKBtn.setFont(font)
        OKBtn.setFixedSize(210, 40)
        OKBtn.clicked.connect(self.okClicked)

        pic = QLabel("", self)
        pic.setFixedSize(400, 700)
        pixmap = QPixmap(self.img_path)
        pixmap = pixmap.scaled(pic.size(), aspectRatioMode=Qt.KeepAspectRatio)
        pic.setPixmap(pixmap)

        vbox = QVBoxLayout()
        vbox.addWidget(pic)
        vbox.addWidget(OKBtn, Qt.AlignCenter)
        vbox.setAlignment(Qt.AlignCenter)

        self.setLayout(vbox) 
        self.setWindowTitle('Розпізнане зображеня')
        self.setModal(True)


    def okClicked(self, event):
        self.accept()

    @staticmethod
    def show(parent, img_path):
        dialog = QPicDialog(parent, img_path)
        dialog.exec_()