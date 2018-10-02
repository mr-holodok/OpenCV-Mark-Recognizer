import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QListView, QFileDialog,
    QHBoxLayout, QVBoxLayout, QApplication, QLayout, QFrame, QSplitter, QToolTip)
from PyQt5.QtGui import (QIcon, QPixmap, QFont, QStandardItemModel, QStandardItem)
from PyQt5.QtCore import (Qt, QModelIndex, QRect, QItemSelectionModel)
from Recognizer import (Recognizer, ImageRotator)
from ResultSaver import (ResultSaver, datetime)
from QPicDialog import QPicDialog
import os


class AppUI(QWidget):
    """class that represent UI of OMR, using PyQt5"""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.pathList = []
        self.rgzr = Recognizer()


    def initUI(self):
        font = QFont()
        font.setPointSize(12)

        QToolTip.setFont(font)

        addBtn = QPushButton(" Додати зображення ", self)
        addBtn.setFont(font)
        addBtn.setFixedSize(210, 40)
        addBtn.clicked.connect(self.showDialog)
        
        recognBtn = QPushButton("Розпізнати", self)
        recognBtn.setFont(font)
        recognBtn.setFixedSize(210, 40)
        recognBtn.clicked.connect(self.recognize)

        delBtn = QPushButton(self)
        delBtn.setFixedSize(50,50)
        delBtn.setIcon(QIcon("images/delete.png"))
        delBtn.clicked.connect(self.deleteImg)
        delBtn.setToolTip("Видалити зображення з набору")

        rotateBtn = QPushButton(self)
        rotateBtn.setFixedSize(50,50)
        rotateBtn.setIcon(QIcon("images/rotate.png"))
        rotateBtn.clicked.connect(self.rotateImg)
        rotateBtn.setToolTip("Повернути зображення")

        self.pic = QLabel("", self)
        self.pic.setFixedSize(330, 580)
        self.pic.setFrameShape(QFrame.Panel)

        lbl = QLabel(" Список зображень: ")
        lbl.setFont(font)
        lbl.setFixedSize(300,40)

        self.picList = QListView(self)
        self.picList.setFixedSize(300, 300)
        self.picList.clicked.connect(self.itemClicked)

        self.listModel = QStandardItemModel(self.picList)
        self.picList.setModel(self.listModel)

        splitter = QSplitter(Qt.Vertical)

        vbtnbox = QVBoxLayout()
        vbtnbox.addStretch(1)
        vbtnbox.addWidget(addBtn)
        vbtnbox.addWidget(recognBtn)
        vbtnbox.setAlignment(Qt.AlignCenter)

        vlistbox = QVBoxLayout()
        vlistbox.addStretch(1)
        vlistbox.addWidget(lbl)
        vlistbox.addWidget(self.picList)

        hbtnbox = QHBoxLayout()
        hbtnbox.addWidget(delBtn)
        hbtnbox.addWidget(rotateBtn)

        vimgbox = QVBoxLayout()
        vimgbox.addWidget(self.pic)
        vimgbox.addLayout(hbtnbox)

        vbox = QVBoxLayout()
        vbox.addLayout(vlistbox)
        vbox.addLayout(vbtnbox)

        hbox = QHBoxLayout()
        #hbox.addWidget(self.pic, alignment=Qt.AlignLeft)
        hbox.addLayout(vimgbox)
        hbox.addWidget(splitter)
        hbox.addLayout(vbox)
        
        self.setLayout(hbox)    
        
        self.setGeometry(300, 300, 600, 600)
        self.setWindowTitle('OMR')    
        self.show()


    def showDialog(self):
        fname = QFileDialog.getOpenFileNames(self, 'Open file', filter='*.jpg')
        if fname[0]:
            for path in fname[0]:
                if path not in self.pathList:
                    self.pathList.append(path)
                    self.addItemToList(path[path.rfind('/')+1:])
            self.sltdImgIndx = len(self.pathList) - 1
            path = self.pathList[-1]
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(self.pic.size(), aspectRatioMode=Qt.KeepAspectRatio)
            self.pic.setPixmap(pixmap)
    
                    
    def addItemToList(self, path):
        item = QStandardItem(path)
        item.setEditable(False)
        self.listModel.appendRow(item)


    def itemClicked(self, sender):
        indx = sender.row()
        self.sltdImgIndx = indx
        pixmap = QPixmap(self.pathList[indx])
        pixmap = pixmap.scaled(self.pic.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.pic.setPixmap(pixmap)


    def recognize(self):
        fname = QFileDialog.getSaveFileName(self, "Оберіть розташування та назву для файла", filter='*.txt')
        if fname[0]:
            saver = ResultSaver(fname[0])
            for img_path in self.pathList:
                answ, image_path= self.rgzr.recognize(img_path)
                saver.write(answ, img_path[img_path.rfind('/'):])
                QPicDialog.show(self, image_path)
                os.remove(image_path)

                

    def deleteImg(self):
        if self.pathList:
            indx = self.sltdImgIndx
            self.pathList.remove(self.pathList[indx])
            self.listModel.removeRow(indx)
            if len(self.pathList) > 0:
                self.sltdImgIndx = 0
                pixmap = QPixmap(self.pathList[0])
                pixmap = pixmap.scaled(self.pic.size(), aspectRatioMode=Qt.KeepAspectRatio)
                self.pic.setPixmap(pixmap)
            else: 
                self.sltdImgIndx = None
                self.pic.clear()


    def rotateImg(self):
        new_path = ImageRotator.rotate(self.pathList[self.sltdImgIndx])
        pixmap = QPixmap(new_path)
        pixmap = pixmap.scaled(self.pic.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.pic.setPixmap(pixmap)
        item = QStandardItem(new_path[new_path.rfind('/')+1:])
        item.setEditable(False)
        self.listModel.setItem(self.sltdImgIndx, item)
        self.pathList[self.sltdImgIndx] = new_path



if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = AppUI()
    sys.exit(app.exec_())
