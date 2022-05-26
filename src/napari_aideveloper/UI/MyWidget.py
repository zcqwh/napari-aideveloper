
#from PyQt5 import QtWidgets,QtCore
from qtpy import QtWidgets,QtCore

class MyTable(QtWidgets.QTableWidget):
    dropped = QtCore.Signal(list) 
    def __init__(self, *args, **kwargs):
        super(MyTable, self).__init__( *args, **kwargs)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        #self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.drag_item = None
        self.drag_row = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.drag_item = None
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))  
            self.dropped.emit(links) #发射信号
            
        else:
            event.ignore()       

    def startartDrag(self, supportedActions):
        super(MyTable, self).startDrag(supportedActions)
        self.drag_item = self.currentItem()
        self.drag_row = self.row(self.drag_item)


class MySpinBox(QtWidgets.QSpinBox):
    #Disable wheelevent for spinbox
    #Replace all SpinBox with SpinBox
    def wheelEvent(self, event):
        event.ignore()

class MyDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    #Disable wheelevent for doublespinbox
    #Replace all DoubleSpinBox with DoubleSpinBox
    def wheelEvent(self, event):
        event.ignore()
