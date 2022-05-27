"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import ast
import copy
import json
import os
import platform
import sys
import time
import traceback
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWGRP, S_IWOTH, S_IWRITE

import cv2
import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tensorflow as tf
from qtpy import QtCore, QtGui, QtWidgets, uic
from .UI import MyWidget
from tensorflow.keras import backend as K
from tensorflow.keras.models import (load_model, model_from_config,
                                     model_from_json)
from tensorflow.keras.utils import to_categorical

from . import aid_bin, aid_dl, aid_img, aid_start, model_zoo

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


VERSION = "0.3.0" #Python 3.9.9 Version
model_zoo_version = model_zoo.__version__()

tooltips = aid_start.get_tooltips()

from tensorflow.python.client import device_lib

dir_root = os.path.dirname(__file__)#ask the module for its origin
sys.path.append(os.path.join(dir_root,"UI")) # For finding CustomWidget
dir_settings = os.path.join(dir_root,"aid_settings.json")#dir to settings
Default_dict = aid_start.get_default_dict(dir_settings)

devices = device_lib.list_local_devices()
device_types = [devices[i].device_type for i in range(len(devices))]

#Get the number  of CPU cores and GPUs
cpu_nr = os.cpu_count()
gpu_nr = device_types.count("GPU")
# =============================================================================
# print("Nr. of CPUs detected: "+str(cpu_nr))
# print("Nr. of GPUs detected: "+str(gpu_nr))
#
# print("List of device(s):")
# print("------------------------")
# for i in range(len(devices)):
#     print("Device "+str(i)+": "+devices[i].name)
#     print("Device type: "+devices[i].device_type)
#     print("Device description: "+devices[i].physical_device_desc)
#     print("------------------------")
# =============================================================================

#Split CPU and GPU into two lists of devices
devices_cpu = []
devices_gpu = []
for dev in devices:
    if dev.device_type=="CPU":
        devices_cpu.append(dev)
    elif dev.device_type=="GPU":
        devices_gpu.append(dev)
    else:
        print("Unknown device type:"+str(dev)+"\n")



try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class MyPopup(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)


class MyTable(QtWidgets.QTableWidget):
# =============================================================================
#     
# =============================================================================
    dropped = QtCore.Signal(list)

    def __init__(self,  rows, columns, parent):
        super().__init__(rows, columns, parent)
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
            self.dropped.emit(links)
        else:
            event.ignore()

    def startDrag(self, supportedActions):
        super().startDrag(supportedActions)
        self.drag_item = self.currentItem()
        self.drag_row = self.row(self.drag_item)

class SpinBox(QtWidgets.QSpinBox):
    #Disable wheelevent for spinbox
    #Replace all SpinBox with SpinBox
    def wheelEvent(self, event):
        event.ignore()

class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    #Disable wheelevent for doublespinbox
    #Replace all DoubleSpinBox with DoubleSpinBox
    def wheelEvent(self, event):
        event.ignore()


class Worker(QtCore.QRunnable):
    '''
    Code inspired/copied from: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''
    def __init__(self, fn, *args, **kwargs):
        super().__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['history_callback'] = self.signals.history
# =============================================================================
# 
# =============================================================================
    @QtCore.Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class WorkerSignals(QtCore.QObject):
    '''
    Code inspired from here: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/

    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    history
        `dict` containing keras model history.history resulting from .fit
    '''
# =============================================================================
#     finished = QtCore.pyqtSignal()
#     error = QtCore.pyqtSignal(tuple)
#     result = QtCore.pyqtSignal(object)
#     progress = QtCore.pyqtSignal(int)
#     history = QtCore.pyqtSignal(dict)
# =============================================================================
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)
    history = QtCore.Signal(dict)


class Fitting_Ui(QtWidgets.QWidget):

    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(850, 786)
        Form.setStyleSheet(open(os.path.join(dir_root,"art","styles","00_base.qss")).read())

        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_4_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_4_pop.setObjectName("verticalLayout_4_pop")
        self.horizontalLayout_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_pop.setObjectName("horizontalLayout_pop")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableWidget_HistoryInfo_pop = QtWidgets.QTableWidget(self.groupBox)
        self.tableWidget_HistoryInfo_pop.setObjectName("tableWidget_HistoryInfo_pop")
        self.tableWidget_HistoryInfo_pop.setColumnCount(7)
        self.tableWidget_HistoryInfo_pop.setRowCount(0)
        self.gridLayout_5.addWidget(self.tableWidget_HistoryInfo_pop, 0, 0, 1, 1)
        self.horizontalLayout_pop.addWidget(self.groupBox)
        self.verticalLayout_2_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_2_pop.setObjectName("verticalLayout_2_pop")
        self.pushButton_UpdatePlot_pop = QtWidgets.QPushButton(Form)
        self.pushButton_UpdatePlot_pop.setObjectName("pushButton_UpdatePlot_pop")
        self.verticalLayout_2_pop.addWidget(self.pushButton_UpdatePlot_pop)
        self.checkBox_realTimePlotting_pop = QtWidgets.QCheckBox(Form)
        self.checkBox_realTimePlotting_pop.setChecked(True)
        self.checkBox_realTimePlotting_pop.setObjectName("checkBox_realTimePlotting_pop")
        self.verticalLayout_2_pop.addWidget(self.checkBox_realTimePlotting_pop)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_realTimeEpochs_pop = QtWidgets.QLabel(Form)
        self.label_realTimeEpochs_pop.setObjectName("label_realTimeEpochs_pop")
        self.horizontalLayout.addWidget(self.label_realTimeEpochs_pop)
        self.spinBox_realTimeEpochs = QtWidgets.QSpinBox(Form)
        self.spinBox_realTimeEpochs.setMinimum(1)
        self.spinBox_realTimeEpochs.setMaximum(9999999)
        self.spinBox_realTimeEpochs.setProperty("value", 250)
        self.spinBox_realTimeEpochs.setObjectName("spinBox_realTimeEpochs")
        self.horizontalLayout.addWidget(self.spinBox_realTimeEpochs)
        self.verticalLayout_2_pop.addLayout(self.horizontalLayout)
        self.horizontalLayout_pop.addLayout(self.verticalLayout_2_pop)
        self.verticalLayout_4_pop.addLayout(self.horizontalLayout_pop)
        self.verticalLayout_3_pop = QtWidgets.QVBoxLayout()
        self.verticalLayout_3_pop.setObjectName("verticalLayout_3_pop")
        self.widget_pop = pg.GraphicsLayoutWidget(Form) #self.widget_pop = QtWidgets.QWidget(Form)
        self.widget_pop.setMinimumSize(771, 331)
        self.widget_pop.setObjectName("widget_pop")
        self.verticalLayout_3_pop.addWidget(self.widget_pop)
        self.splitter_pop = QtWidgets.QSplitter(Form)
        self.splitter_pop.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_pop.setObjectName("splitter_pop")
        self.groupBox_FittingInfo_pop = QtWidgets.QGroupBox(self.splitter_pop)
        self.groupBox_FittingInfo_pop.setObjectName("groupBox_FittingInfo_pop")
        self.gridLayout_2_pop = QtWidgets.QGridLayout(self.groupBox_FittingInfo_pop)
        self.gridLayout_2_pop.setObjectName("gridLayout_2_pop")
        self.progressBar_Fitting_pop = QtWidgets.QProgressBar(self.groupBox_FittingInfo_pop)
        self.progressBar_Fitting_pop.setProperty("value", 24)
        self.progressBar_Fitting_pop.setObjectName("progressBar_Fitting_pop")
        self.gridLayout_2_pop.addWidget(self.progressBar_Fitting_pop, 0, 0, 1, 1)
        self.textBrowser_FittingInfo = QtWidgets.QTextBrowser(self.groupBox_FittingInfo_pop)
        self.textBrowser_FittingInfo.setObjectName("textBrowser_FittingInfo")
        self.gridLayout_2_pop.addWidget(self.textBrowser_FittingInfo, 1, 0, 1, 1)
        self.horizontalLayout_saveClearText_pop = QtWidgets.QHBoxLayout()
        self.horizontalLayout_saveClearText_pop.setObjectName("horizontalLayout_saveClearText_pop")
        self.pushButton_saveTextWindow_pop = QtWidgets.QPushButton(self.groupBox_FittingInfo_pop)
        self.pushButton_saveTextWindow_pop.setObjectName("pushButton_saveTextWindow_pop")
        self.horizontalLayout_saveClearText_pop.addWidget(self.pushButton_saveTextWindow_pop)
        self.pushButton_clearTextWindow_pop = QtWidgets.QPushButton(self.groupBox_FittingInfo_pop)
        self.pushButton_clearTextWindow_pop.setObjectName("pushButton_clearTextWindow_pop")
        self.horizontalLayout_saveClearText_pop.addWidget(self.pushButton_clearTextWindow_pop)
        self.gridLayout_2_pop.addLayout(self.horizontalLayout_saveClearText_pop, 2, 0, 1, 1)
        self.groupBox_ChangeModel_pop = QtWidgets.QGroupBox(self.splitter_pop)
        self.groupBox_ChangeModel_pop.setEnabled(True)
        self.groupBox_ChangeModel_pop.setCheckable(False)
        self.groupBox_ChangeModel_pop.setObjectName("groupBox_ChangeModel_pop")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_ChangeModel_pop)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_expt_imgProc_pop = QtWidgets.QGroupBox(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_expt_imgProc_pop.sizePolicy().hasHeightForWidth())
        self.groupBox_expt_imgProc_pop.setSizePolicy(sizePolicy)
        self.groupBox_expt_imgProc_pop.setObjectName("groupBox_expt_imgProc_pop")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_expt_imgProc_pop)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_icon_padding = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_icon_padding.sizePolicy().hasHeightForWidth())
        self.label_icon_padding.setSizePolicy(sizePolicy)
        self.label_icon_padding.setMinimumSize(21, 21)
        self.label_icon_padding.setMaximumSize(21, 21)
        self.label_icon_padding.setText("")
        self.label_icon_padding.setPixmap(QtGui.QPixmap(os.path.join(dir_root, "art","Icon","padding.png")))
        self.label_icon_padding.setScaledContents(True)
        self.label_icon_padding.setAlignment(QtCore.Qt.AlignCenter)
        self.label_icon_padding.setObjectName("label_icon_padding")
        self.gridLayout_2.addWidget(self.label_icon_padding, 0, 0, 1, 1)
        self.label_paddingMode_pop = QtWidgets.QLabel(self.groupBox_expt_imgProc_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_paddingMode_pop.sizePolicy().hasHeightForWidth())
        self.label_paddingMode_pop.setSizePolicy(sizePolicy)
        self.label_paddingMode_pop.setObjectName("label_paddingMode_pop")
        self.gridLayout_2.addWidget(self.label_paddingMode_pop, 0, 1, 1, 1)
        self.comboBox_paddingMode_pop = QtWidgets.QComboBox(self.groupBox_expt_imgProc_pop)
        self.comboBox_paddingMode_pop.setEnabled(True)
        self.comboBox_paddingMode_pop.setObjectName("comboBox_paddingMode_pop")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.comboBox_paddingMode_pop.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_paddingMode_pop, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.groupBox_expt_imgProc_pop, 0, 0, 1, 5)
        self.groupBox_system_pop = QtWidgets.QGroupBox(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_system_pop.sizePolicy().hasHeightForWidth())
        self.groupBox_system_pop.setSizePolicy(sizePolicy)
        self.groupBox_system_pop.setObjectName("groupBox_system_pop")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_system_pop)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.spinBox_NrEpochs = QtWidgets.QSpinBox(self.groupBox_system_pop)
        self.spinBox_NrEpochs.setMinimum(1)
        self.spinBox_NrEpochs.setMaximum(999999999)
        self.spinBox_NrEpochs.setObjectName("spinBox_NrEpochs")
        self.gridLayout_3.addWidget(self.spinBox_NrEpochs, 0, 2, 1, 1)
        self.label_saveMetaEvery = QtWidgets.QLabel(self.groupBox_system_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_saveMetaEvery.sizePolicy().hasHeightForWidth())
        self.label_saveMetaEvery.setSizePolicy(sizePolicy)
        self.label_saveMetaEvery.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_saveMetaEvery.setObjectName("label_saveMetaEvery")
        self.gridLayout_3.addWidget(self.label_saveMetaEvery, 0, 3, 1, 1)
        self.spinBox_saveMetaEvery = QtWidgets.QSpinBox(self.groupBox_system_pop)
        self.spinBox_saveMetaEvery.setMinimum(1)
        self.spinBox_saveMetaEvery.setMaximum(999999)
        self.spinBox_saveMetaEvery.setProperty("value", 30)
        self.spinBox_saveMetaEvery.setObjectName("spinBox_saveMetaEvery")
        self.gridLayout_3.addWidget(self.spinBox_saveMetaEvery, 0, 4, 1, 1)
        self.label_Crop_NrEpochs_pop = QtWidgets.QLabel(self.groupBox_system_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_Crop_NrEpochs_pop.sizePolicy().hasHeightForWidth())
        self.label_Crop_NrEpochs_pop.setSizePolicy(sizePolicy)
        self.label_Crop_NrEpochs_pop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_Crop_NrEpochs_pop.setObjectName("label_Crop_NrEpochs_pop")
        self.gridLayout_3.addWidget(self.label_Crop_NrEpochs_pop, 0, 1, 1, 1)
        self.label_icon_epochs = QtWidgets.QLabel(self.groupBox_system_pop)
        self.label_icon_epochs.setMinimumSize(20, 20)
        self.label_icon_epochs.setMaximumSize(20, 20)
        self.label_icon_epochs.setText("")
        self.label_icon_epochs.setPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","Icon","nr_epochs.png")))
        self.label_icon_epochs.setScaledContents(True)
        self.label_icon_epochs.setObjectName("label_icon_epochs")
        self.gridLayout_3.addWidget(self.label_icon_epochs, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_system_pop, 1, 0, 1, 5)
        self.splitter = QtWidgets.QSplitter(self.groupBox_ChangeModel_pop)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.pushButton_showModelSumm_pop = QtWidgets.QPushButton(self.splitter)
        self.pushButton_showModelSumm_pop.setObjectName("pushButton_showModelSumm_pop")
        self.pushButton_saveModelSumm_pop = QtWidgets.QPushButton(self.splitter)
        self.pushButton_saveModelSumm_pop.setObjectName("pushButton_saveModelSumm_pop")
        self.gridLayout.addWidget(self.splitter, 2, 0, 1, 5)
        self.checkBox_ApplyNextEpoch = QtWidgets.QCheckBox(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_ApplyNextEpoch.sizePolicy().hasHeightForWidth())
        self.checkBox_ApplyNextEpoch.setSizePolicy(sizePolicy)
        self.checkBox_ApplyNextEpoch.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox_ApplyNextEpoch.setAutoFillBackground(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root, "art","Icon","thumb.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.checkBox_ApplyNextEpoch.setIconSize(QtCore.QSize(20, 20))
        self.checkBox_ApplyNextEpoch.setIcon(icon)
        self.checkBox_ApplyNextEpoch.setTristate(False)
        self.checkBox_ApplyNextEpoch.setObjectName("checkBox_ApplyNextEpoch")
        self.gridLayout.addWidget(self.checkBox_ApplyNextEpoch, 3, 0, 1, 1)
        self.checkBox_saveEpoch_pop = QtWidgets.QCheckBox(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_saveEpoch_pop.sizePolicy().hasHeightForWidth())
        self.checkBox_saveEpoch_pop.setSizePolicy(sizePolicy)
        self.checkBox_saveEpoch_pop.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(dir_root, "art","Icon","save_epoch.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.checkBox_saveEpoch_pop.setIconSize(QtCore.QSize(20, 20))
        self.checkBox_saveEpoch_pop.setIcon(icon1)
        self.checkBox_saveEpoch_pop.setObjectName("checkBox_saveEpoch_pop")
        self.gridLayout.addWidget(self.checkBox_saveEpoch_pop, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 2, 1, 1)
        self.pushButton_Pause_pop = QtWidgets.QPushButton(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Pause_pop.sizePolicy().hasHeightForWidth())
        self.pushButton_Pause_pop.setSizePolicy(sizePolicy)
        self.pushButton_Pause_pop.setMinimumSize(60, 40)
        self.pushButton_Pause_pop.setMaximumSize(60, 40)
        self.pushButton_Pause_pop.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join(dir_root, "art","Icon","pause.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Pause_pop.setIcon(icon2)
        self.pushButton_Pause_pop.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_Pause_pop.setObjectName("pushButton_Pause_pop")
        self.gridLayout.addWidget(self.pushButton_Pause_pop, 3, 3, 1, 1)
        self.pushButton_Stop_pop = QtWidgets.QPushButton(self.groupBox_ChangeModel_pop)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Stop_pop.sizePolicy().hasHeightForWidth())
        self.pushButton_Stop_pop.setSizePolicy(sizePolicy)
        self.pushButton_Stop_pop.setMinimumSize(60, 40)
        self.pushButton_Stop_pop.setMaximumSize(60, 40)
        self.pushButton_Stop_pop.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join(dir_root, "art","Icon","stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Stop_pop.setIcon(icon3)
        self.pushButton_Stop_pop.setIconSize(QtCore.QSize(18, 18))
        self.pushButton_Stop_pop.setObjectName("pushButton_Stop_pop")
        self.gridLayout.addWidget(self.pushButton_Stop_pop, 3, 4, 1, 1)
        self.verticalLayout_3_pop.addWidget(self.splitter_pop)
        self.verticalLayout_4_pop.addLayout(self.verticalLayout_3_pop)
        self.gridLayout_4.addLayout(self.verticalLayout_4_pop, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

          #####################Some manual settings##############################
        #######################################################################
        ###########################Variables###################################
        self.Histories = [] #List container for the fitting histories, that are produced by the keras.fit function that is controlled by this popup
        self.RealTime_Acc,self.RealTime_ValAcc,self.RealTime_Loss,self.RealTime_ValLoss = [],[],[],[]
        self.RealTime_OtherMetrics = {} #provide dictionary where AID can save all other metrics in case there are some (like precision...)
        self.X_batch_aug = []#list for storing augmented image, created by some parallel processes
        self.threadpool_quad = QtCore.QThreadPool()#Threadpool for image augmentation
        self.threadpool_quad.setMaxThreadCount(4)#Maximum 4 threads
        self.threadpool_quad_count = 0 #count nr. of threads in queue;
        self.clr_settings = {} #variable to store step_size and gamma, will be filled with information when starting to fit
        self.optimizer_settings = {} #dict to store advanced optimizer settings

        self.epoch_counter = 0 #Counts the nr. of epochs

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Metrics keys"))
        self.pushButton_UpdatePlot_pop.setText(_translate("Form", "Update Plot"))
        self.checkBox_realTimePlotting_pop.setToolTip(_translate("Form", "<html><head/><body><p>Add for each curve a rolling median curve, which uses a window size of 10</p></body></html>"))
        self.checkBox_realTimePlotting_pop.setText(_translate("Form", "Real time plotting"))
        self.label_realTimeEpochs_pop.setText(_translate("Form", "Nr. of epochs for RT"))
        self.groupBox_FittingInfo_pop.setTitle(_translate("Form", "Fitting Info"))
        self.pushButton_saveTextWindow_pop.setText(_translate("Form", "Save text "))
        self.pushButton_clearTextWindow_pop.setToolTip(_translate("Form", "Clear the text window (fitting info)"))
        self.pushButton_clearTextWindow_pop.setText(_translate("Form", "Clear text"))
        self.groupBox_ChangeModel_pop.setTitle(_translate("Form", "Change fitting parameters"))
        self.groupBox_expt_imgProc_pop.setTitle(_translate("Form", "Image processing"))
        self.label_paddingMode_pop.setText(_translate("Form", "Padding mode"))
        self.comboBox_paddingMode_pop.setToolTip(_translate("Form", "By default, the padding mode is \"constant\", which means that zeros are padded.\n"
"\"edge\": Pads with the edge values of array.\n"
"\"linear_ramp\": Pads with the linear ramp between end_value and the array edge value.\n"
"\"maximum\": Pads with the maximum value of all or part of the vector along each axis.\n"
"\"mean\": Pads with the mean value of all or part of the vector along each axis.\n"
"\"median\": Pads with the median value of all or part of the vector along each axis.\n"
"\"minimum\": Pads with the minimum value of all or part of the vector along each axis.\n"
"\"reflect\": Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.\n"
"\"symmetric\": Pads with the reflection of the vector mirrored along the edge of the array.\n"
"\"wrap\": Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.\n"
"Text copied from https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html"))
        self.comboBox_paddingMode_pop.setItemText(0, _translate("Form", "constant"))
        self.comboBox_paddingMode_pop.setItemText(1, _translate("Form", "edge"))
        self.comboBox_paddingMode_pop.setItemText(2, _translate("Form", "linear_ramp"))
        self.comboBox_paddingMode_pop.setItemText(3, _translate("Form", "maximum"))
        self.comboBox_paddingMode_pop.setItemText(4, _translate("Form", "mean"))
        self.comboBox_paddingMode_pop.setItemText(5, _translate("Form", "median"))
        self.comboBox_paddingMode_pop.setItemText(6, _translate("Form", "minimum"))
        self.groupBox_system_pop.setTitle(_translate("Form", "Training"))
        self.spinBox_NrEpochs.setToolTip(_translate("Form", "Total number of training iterations"))
        self.label_saveMetaEvery.setText(_translate("Form", "Save meta every (sec)"))
        self.label_Crop_NrEpochs_pop.setToolTip(_translate("Form", "Total number of training iterations"))
        self.label_Crop_NrEpochs_pop.setText(_translate("Form", "Nr. epochs"))
        self.pushButton_showModelSumm_pop.setText(_translate("Form", "Show model summary"))
        self.pushButton_saveModelSumm_pop.setText(_translate("Form", "Save model summary"))
        self.checkBox_ApplyNextEpoch.setToolTip(_translate("Form", "Changes made in this window will be applied at the next epoch"))
        self.checkBox_ApplyNextEpoch.setText(_translate("Form", "Apply at next epoch"))
        self.checkBox_saveEpoch_pop.setToolTip(_translate("Form", "Save the model, when the current epoch is done"))
        self.checkBox_saveEpoch_pop.setText(_translate("Form", "Save epoch"))
        self.pushButton_Pause_pop.setToolTip(_translate("Form", "Pause fitting, push this button again to continue"))
        self.pushButton_Stop_pop.setToolTip(_translate("Form", "Stop fitting entirely, Close this window manually, after the progressbar shows 100%"))




class AIDeveloper(QtWidgets.QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        uic.loadUi(os.path.join(dir_root,"UI","napari_AID.ui"),self)
        
# =============================================================================
#         Link functions
# =============================================================================
        ####### table_dragdrop #######
        for i in [1]:
            self.table_dragdrop.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.Interactive)
            
        for i in [0,2,3,4,5,6,7,8,9,10,11]:
            self.table_dragdrop.horizontalHeader().setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeToContents)
        
        self.table_dragdrop.setAcceptDrops(True)
        self.table_dragdrop.setDragEnabled(True)
        self.table_dragdrop.dropped.connect(self.dataDropped)
        self.table_dragdrop.clicked.connect(self.item_click)
        self.table_dragdrop.doubleClicked.connect(self.item_dclick)
        self.table_dragdrop.itemChanged.connect(self.dataOverviewOn_OnChange)
        self.table_dragdrop.itemChanged.connect(self.uncheck_if_zero)
        self.table_dragdrop.horizontalHeader().sectionClicked.connect(self.select_all)
        self.table_dragdrop.resizeRowsToContents()
        
        self.btn_show.clicked.connect(self.action_show_example_imgs)
        
        ####### Load model from model_zoo #######
        self.predefined_models = model_zoo.get_predefined_models()
        #self.predefined_models.sort()
        self.predefined_models = ["None"] + self.predefined_models
        self.comboBox_ModelSelection.addItems(self.predefined_models)
        
        ####### model training #######
        self.radioButton_LoadRestartModel.clicked.connect(self.action_preview_model)
        self.radioButton_LoadContinueModel.clicked.connect(self.action_preview_model)
        self.pushButton_modelname.clicked.connect(self.action_set_modelpath_and_name)
        self.pushButton_FitModel.clicked.connect(lambda: self.action_initialize_model(duties="initialize_train"))


        ############################Variables##################################
        #######################################################################
        #Initilaize some variables which are lateron filled in the program
        self.w = None #Initialize a variable for a popup window
        self.threadpool = QtCore.QThreadPool()
        self.threadpool_single = QtCore.QThreadPool()
        self.threadpool_single.setMaxThreadCount(1)
        self.threadpool_single_queue = 0 #count nr. of threads in queue;

        #self.threadpool_single = QtCore.QThread()
        self.fittingpopups = []  #This app will be designed to allow training of several models ...
        self.fittingpopups_ui = [] #...simultaneously (threading). The info of each model is appended to a list

        self.popupcounter = 0
        self.colorsQt = 10*['Crimson','yellow','DodgerBlue','cyan','Violet','green','gray','darkRed','darkYellow','darkBlue','darkCyan','darkMagenta','darkGreen','darkGray']    #Some colors which are later used for different subpopulations
        self.model_keras = None #Variable for storing Keras model
        self.model_keras_path = None
        self.load_model_path = None
        self.loaded_history = None #Variable for storing a loaded history file (for display on History-Tab)
        self.loaded_para = None #Variable for storing a loaded Parameters-file (for display on History-Tab)
        self.plt1 = None #Used for the popup window to display hist and scatter of single experiments
        self.plt2 = None #Used for the history-tab to show accuracy of loaded history files
        self.plt_cm = [] #Used to show images from the interactive Confusion matrix
        self.model_2_convert = None #Variable to store the path to a chosen model (for converting to .nnet)
        self.ram = dict() #Variable to store data if Option "Data to RAM is enabled"
        self.ValidationSet = None
        self.Metrics = dict()
        self.clr_settings = {}
        self.clr_settings["step_size"] = 8 #Number of epochs to fulfill half a cycle
        self.clr_settings["gamma"] = 0.99995 #gamma factor for Exponential decrease method (exp_range)
        self.optimizer_settings = aid_dl.get_optimizer_settings() #the full set of optimizer settings is saved in this variable and might be changed usiung pushButton_optimizer

        #self.clip = QtGui.QApplication.clipboard() #This is how one defines a clipboard variable; one can put text on it via:#self.clip.setText("SomeText")
        self.new_peaks = [] #list to store used defined peaks
        #######################################################################
        #######################################################################
        self.norm_methods = Default_dict["norm_methods"]


    def dataDropped(self, l):

        #Iterate over l and check if it is a folder or a file (directory)
        isfile = [os.path.isfile(str(url)) for url in l]

        ind_true = np.where(np.array(isfile)==True)[0]
        filenames = list(np.array(l)[ind_true]) #select the indices that are valid
        filenames = [x for x in filenames if x.endswith(".rtdc")]

        fileinfo = []
        for i in range(len(filenames)):
            rtdc_path = filenames[i]

            try:
                failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
                if failed:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(str(rtdc_ds))
                    msg.setWindowTitle("Error occurred during loading file")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                features = list(rtdc_ds["events"].keys())
                #Make sure that there is "images", "pos_x" and "pos_y" available
                if "image" in features and "pos_x" in features and "pos_y" in features:
                    nr_images = rtdc_ds["events"]["image"].len()
                    pix = rtdc_ds.attrs["imaging:pixel size"]
                    xtra_in_available = len(rtdc_ds.keys())>2 #Is True, only if there are more than 2 elements.
                    fileinfo.append({"rtdc_ds":rtdc_ds,"rtdc_path":rtdc_path,"features":features,"nr_images":nr_images,"pix":pix,"xtra_in":xtra_in_available})
                else:
                    missing = []
                    for feat in ["image","pos_x","pos_y"]:
                        if feat not in features:
                            missing.append(feat)
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("Essential feature(s) are missing in data-set")
                    msg.setDetailedText("Data-set: "+rtdc_path+"\nis missing "+str(missing))
                    msg.setWindowTitle("Missing essential features")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()

            except Exception as e:
                print(e)


        for rowNumber in range(len(fileinfo)):#for url in l:
            url = fileinfo[rowNumber]["rtdc_path"]
            #add to table
            rowPosition = self.table_dragdrop.rowCount()
            self.table_dragdrop.insertRow(rowPosition)
            
            columnPosition = 0 #Del
            btn_delete = QtWidgets.QPushButton(self.table_dragdrop)
            btn_delete.setMaximumSize(QtCore.QSize(30, 30))
            icon_del = QtGui.QIcon()
            icon_del.addPixmap(QtGui.QPixmap(":/icon/icon/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            btn_delete.setIcon(icon_del)
            self.table_dragdrop.setCellWidget(rowPosition, columnPosition, btn_delete) 
            #self.table_dragdrop.resizeRowsToContents()
            btn_delete.clicked.connect(self.delete_item)  
            
            
            columnPosition = 1 #File
            line = QtWidgets.QTableWidgetItem()
            line.setText(url)
            line.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
            line.setToolTip(url)

            self.table_dragdrop.setItem(rowPosition, columnPosition, line)
            self.table_dragdrop.setColumnWidth(columnPosition,100)
            

            columnPosition = 2 #Class
            comboBox = QtWidgets.QComboBox(self.table_dragdrop)
            class_items =["0","1","2","3","4","5","6","7","8","9"]
            comboBox.addItems(class_items)
            comboBox.setStyleSheet("QComboBox {"
                                   "combobox-popup: 0;}"

                                    "QComboBox:drop-down {"
                                    "width:20px; "
                                    "subcontrol-position: right center; "  # 位置
                                    "subcontrol-origin: padding;}\n"  # 对齐方式
                                    )

            comboBox.currentIndexChanged.connect(self.dataOverviewOn)
            self.table_dragdrop.setCellWidget(rowPosition, columnPosition, comboBox)

            for columnPosition in [3,4]: #T/V
                #for each item, also create 2 checkboxes (train/valid)
                item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
                item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
                item.setCheckState(QtCore.Qt.Unchecked)
                self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 5 #Show plot
            #Place a button which allows to show a plot (scatter, histo...lets see)
            btn = QtWidgets.QPushButton(self.table_dragdrop)
            btn.setMinimumSize(50, 30)
            btn.setMaximumSize(50, 30)
            btn.clicked.connect(self.button_hist)
            btn.setText('Plot')
            self.table_dragdrop.setCellWidget(rowPosition, columnPosition, btn)
            self.table_dragdrop.resizeRowsToContents()

            columnPosition = 6 #Events
            #Place a combobox with the available features
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.DisplayRole, fileinfo[rowNumber]["nr_images"])
            item.setFlags(item.flags() &~QtCore.Qt.ItemIsEnabled &~ QtCore.Qt.ItemIsSelectable )
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 7 #Events/Epoch
            #Field to user-define nr. of cells/epoch
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.EditRole,100)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 8 #pixel
            #Pixel size
            item = QtWidgets.QTableWidgetItem()
            pix = float(fileinfo[rowNumber]["pix"])
            #print(pix)
            item.setData(QtCore.Qt.EditRole,pix)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 9 #Shuffle
            #Should data be shuffled (random?)
            item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
            item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            item.setCheckState(QtCore.Qt.Checked)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 10 #Zoom
            #Zooming factor
            item = QtWidgets.QTableWidgetItem()
            zoom = 1.0
            item.setData(QtCore.Qt.EditRole,zoom)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)

            columnPosition = 11  #Xtra_In
            #Should xtra_data be used?
            item = QtWidgets.QTableWidgetItem()
            xtra_in_available = fileinfo[rowNumber]["xtra_in"]
            if xtra_in_available:
                item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            else:
                item.setFlags( QtCore.Qt.ItemIsUserCheckable )
            item.setCheckState(QtCore.Qt.Unchecked)

            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.table_dragdrop.setItem(rowPosition, columnPosition, item)




    def dataOverviewOn(self):
        if self.groupBox_data_overview.isChecked()==True:
            if self.threadpool_single_queue == 0:
                SelectedFiles = self.items_clicked_no_rtdc_ds()
                self.update_data_overview(SelectedFiles)
                #self.update_data_overview_2(SelectedFiles)


    def dataOverviewOn_OnChange(self,item):
            #When a value is entered in Events/Epoch and enter is hit
            #there is no update of the table called
            if self.groupBox_data_overview.isChecked()==True:
                if self.threadpool_single_queue == 0:
                    rowPosition = item.row()
                    colPosition = item.column()
                    if colPosition==7:#one when using the spinbox (Class),or when entering a new number in "Events/Epoch", the table is not updated.
                        #get the new value
                        nr_cells = self.table_dragdrop.cellWidget(rowPosition, colPosition)
                        if nr_cells==None:
                            return
                        else:
                            SelectedFiles = self.items_clicked_no_rtdc_ds()
                            self.update_data_overview(SelectedFiles)
                            #self.update_data_overview_2(SelectedFiles)


    def items_clicked_no_rtdc_ds(self):
        #This function checks, which data has been checked on table_dragdrop and returns the necessary data
        rowCount = self.table_dragdrop.rowCount()
        #Collect urls to files that are checked
        SelectedFiles = []
        for rowPosition in range(rowCount):
            #get the filename/path
            rtdc_path = str(self.table_dragdrop.item(rowPosition, 1).text())
            #get the index (celltype) of it
            index = int(self.table_dragdrop.cellWidget(rowPosition, 2).currentText())
            #How many Events contains dataset in total?
            nr_events = int(self.table_dragdrop.item(rowPosition, 6).text())
            #how many cells/epoch during training or validation?
            nr_events_epoch = int(self.table_dragdrop.item(rowPosition, 7).text())
            #should the dataset be randomized (shuffled?)
            shuffle = bool(self.table_dragdrop.item(rowPosition, 9).checkState())
            #should the images be zoomed in/out by a factor?
            zoom_factor = float(self.table_dragdrop.item(rowPosition, 10).text())
            #should xtra_data be used for training?
            xtra_in = bool(self.table_dragdrop.item(rowPosition, 11).checkState())

            #is it checked for train?
            cb_t = self.table_dragdrop.item(rowPosition, 3)
            if cb_t.checkState() == QtCore.Qt.Checked and nr_events_epoch>0: #add to training files if the user wants more than 0 images per epoch
                #SelectedFiles.append({"nr_images":nr_events,"class":index,"TrainOrValid":"Train","nr_events":nr_events,"nr_events_epoch":nr_events_epoch})
                SelectedFiles.append({"rtdc_path":rtdc_path,"class":index,"TrainOrValid":"Train","nr_events":nr_events,"nr_events_epoch":nr_events_epoch,"shuffle":shuffle,"zoom_factor":zoom_factor,"xtra_in":xtra_in})

            cb_v = self.table_dragdrop.item(rowPosition, 4)
            if cb_v.checkState() == QtCore.Qt.Checked and nr_events_epoch>0:
                #SelectedFiles.append({"nr_images":nr_events,"class":index,"TrainOrValid":"Valid","nr_events":nr_events,"nr_events_epoch":nr_events_epoch})
                SelectedFiles.append({"rtdc_path":rtdc_path,"class":index,"TrainOrValid":"Valid","nr_events":nr_events,"nr_events_epoch":nr_events_epoch,"shuffle":shuffle,"zoom_factor":zoom_factor,"xtra_in":xtra_in})

        return SelectedFiles



    def update_data_overview(self,SelectedFiles):
        #Check if there are custom class names (determined by user)
        rows = self.tableWidget_Info.rowCount()
        self.classes_custom = [] #by default assume there are no custom classes
        classes_custom_bool = False
        if rows>0:#if >0, then there is already a table existing
            classes,self.classes_custom = [],[]
            for row in range(rows):
                try:
                    class_ = self.tableWidget_Info.item(row,0).text()
                    if class_.isdigit():
                        classes.append(class_)#get the classes
                except:
                    pass
                try:
                    self.classes_custom.append(self.tableWidget_Info.item(row,3).text())#get the classes
                except:
                    pass
            classes = np.unique(classes)
            if len(classes)==len(self.classes_custom):#equal in length
                same = [i for i, j in zip(classes, self.classes_custom) if i == j] #which items are identical?
                if len(same)==0:
                    #apparently there are custom classes! Save them
                    classes_custom_bool = True

        if len(SelectedFiles)==0:#reset the table
            #Table1
            #Prepare a table in tableWidget_Info
            self.tableWidget_Info.setColumnCount(0)
            self.tableWidget_Info.setRowCount(0)
            self.tableWidget_Info.setColumnCount(4)
            header = self.tableWidget_Info.horizontalHeader()
            header_labels = ["Class","Events tot.","Events/Epoch","Name"]
            self.tableWidget_Info.setHorizontalHeaderLabels(header_labels)
            header = self.tableWidget_Info.horizontalHeader()
            for i in range(4):
                header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
                #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
            return
        #Prepare a table in tableWidget_Info
        self.tableWidget_Info.setColumnCount(0)
        self.tableWidget_Info.setRowCount(0)

        indices = [SelectedFiles[i]["class"] for i in range(len(SelectedFiles))]
        self.tableWidget_Info.setColumnCount(4)
        header = self.tableWidget_Info.horizontalHeader()

        nr_ind = len(set(indices)) #each index could occur for train and valid
        nr_rows = 2*nr_ind+2 #add two rows for intermediate headers (Train/Valid)
        self.tableWidget_Info.setRowCount(nr_rows)
        #Wich selected file has the most features?
        header_labels = ["Class","Events tot.","Events/Epoch","Name"]
        self.tableWidget_Info.setHorizontalHeaderLabels(header_labels)
        #self.tableWidget_Info.resizeColumnsToContents()
        header = self.tableWidget_Info.horizontalHeader()
        for i in range(4):
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)

        #Training info
        rowPosition = 0
        self.tableWidget_Info.setSpan(rowPosition, 0, 1, 4)
        item = QtWidgets.QTableWidgetItem("Train. data")
        item.setTextAlignment(QtCore.Qt.AlignHCenter| QtCore.Qt.AlignVCenter)

        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.tableWidget_Info.setItem(rowPosition, 0, item)
        rowPosition += 1
        ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
        ind = np.where(np.array(ind)==True)[0]
        SelectedFiles_train = np.array(SelectedFiles)[ind]
        SelectedFiles_train = list(SelectedFiles_train)
        indices_train = [selectedfile["class"] for selectedfile in SelectedFiles_train]

        classes = np.unique(indices_train)
        if len(classes)==len(self.classes_custom):
            classes_custom_bool = True
        else:
            classes_custom_bool = False

       #display information for each individual class
        for index_ in range(len(classes)):
        #for index in np.unique(indices_train):
            index = classes[index_]
            #put the index in column nr. 0
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole,str(index))
            self.tableWidget_Info.setItem(rowPosition, 0, item)
            #Get the training files of that index
            ind = np.where(indices_train==index)[0]
            SelectedFiles_train_index = np.array(SelectedFiles_train)[ind]
            #Total nr of cells for each class
            nr_events = [int(selectedfile["nr_events"]) for selectedfile in SelectedFiles_train_index]
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole, str(np.sum(nr_events)))
            self.tableWidget_Info.setItem(rowPosition, 1, item)
            nr_events_epoch = [int(selectedfile["nr_events_epoch"]) for selectedfile in SelectedFiles_train_index]
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole, str(np.sum(nr_events_epoch)))
            self.tableWidget_Info.setItem(rowPosition, 2, item)

            item = QtWidgets.QTableWidgetItem()
            if classes_custom_bool==False:
                item.setData(QtCore.Qt.EditRole,str(index))
            else:
                item.setData(QtCore.Qt.EditRole,self.classes_custom[index_])
            self.tableWidget_Info.setItem(rowPosition, 3, item)

            rowPosition += 1

        #Validation info
        self.tableWidget_Info.setSpan(rowPosition, 0, 1, 4)
        item = QtWidgets.QTableWidgetItem("Val. data")
        item.setTextAlignment(QtCore.Qt.AlignHCenter| QtCore.Qt.AlignVCenter)
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.tableWidget_Info.setItem(rowPosition, 0, item)
        rowPosition += 1
        ind = [selectedfile["TrainOrValid"] == "Valid" for selectedfile in SelectedFiles]
        ind = np.where(np.array(ind)==True)[0]
        SelectedFiles_valid = np.array(SelectedFiles)[ind]
        SelectedFiles_valid = list(SelectedFiles_valid)
        indices_valid = [selectedfile["class"] for selectedfile in SelectedFiles_valid]
        #Total nr of cells for each index
        for index in np.unique(indices_valid):
            #put the index in column nr. 0
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole,str(index))
            self.tableWidget_Info.setItem(rowPosition, 0, item)
            #Get the validation files of that index
            ind = np.where(indices_valid==index)[0]
            SelectedFiles_valid_index = np.array(SelectedFiles_valid)[ind]
            nr_events = [int(selectedfile["nr_events"]) for selectedfile in SelectedFiles_valid_index]
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole, str(np.sum(nr_events)))
            self.tableWidget_Info.setItem(rowPosition, 1, item)
            nr_events_epoch = [int(selectedfile["nr_events_epoch"]) for selectedfile in SelectedFiles_valid_index]
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            item.setData(QtCore.Qt.EditRole, str(np.sum(nr_events_epoch)))
            self.tableWidget_Info.setItem(rowPosition, 2, item)
            rowPosition += 1
        self.tableWidget_Info.resizeColumnsToContents()
        self.tableWidget_Info.resizeRowsToContents()

    def delete_item(self,item):
        """
        delete table item and corresponding layers
        """
        buttonClicked = self.sender()
        table = buttonClicked.parent().parent()

        index = table.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        table.removeRow(rowPosition) #remove table item

    def item_click(self,item):
        colPosition = item.column()
        rowPosition = item.row()
        #if Shuffle was clicked (col=8), check if this checkbox is not deactivated
        if colPosition==9:
            if bool(self.table_dragdrop.item(rowPosition, 9).checkState())==False:
                rtdc_path = self.table_dragdrop.item(rowPosition, 1).text()
                rtdc_path = str(rtdc_path)

                failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
                if failed:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(str(rtdc_ds))
                    msg.setWindowTitle("Error occurred during loading file")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return
                nr_images = rtdc_ds["events"]["image"].len()

                columnPosition = 7 #Ev./Ep.
                item = QtWidgets.QTableWidgetItem()
                item.setData(QtCore.Qt.DisplayRole, nr_images)
                item.setFlags(item.flags() &~QtCore.Qt.ItemIsEnabled &~ QtCore.Qt.ItemIsSelectable )
                self.table_dragdrop.setItem(rowPosition, columnPosition, item)
            if bool(self.table_dragdrop.item(rowPosition, 9).checkState())==True:
                #Inspect this table item. If shuffle was checked before, it will be grayed out. Invert normal cell then
                item = self.table_dragdrop.item(rowPosition, 7)
                item.setFlags(item.flags() |QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable )

# =============================================================================
#         if len(self.ram)>0:
#             self.statusbar.showMessage("Make sure to update RAM (->Edit->Data to RAM now) after changing Data-set",2000)
#             self.ram = dict() #clear the ram, since the data was changed
#
# =============================================================================
        self.dataOverviewOn()

        #When data is clicked, always reset the validation set (only important for 'Assess Model'-tab)
        self.ValidationSet = None
        self.Metrics = dict() #Also reset the metrics


    def item_dclick(self, item):
        #Check/Uncheck if item is from column 3(T) or 4()V
        tableitem = self.table_dragdrop.item(item.row(), item.column())
        if item.column() in [3,4]:
            #If the item is unchecked ->check it!
            if tableitem.checkState() == QtCore.Qt.Unchecked:
                tableitem.setCheckState(QtCore.Qt.Checked)
            #else, the other way around
            elif tableitem.checkState() == QtCore.Qt.Checked:
                tableitem.setCheckState(QtCore.Qt.Unchecked)

        #Show example image if item on column 1(file) was dclicked
        if item.column() == 1: # 1 File
            rtdc_path = self.table_dragdrop.item(item.row(), item.column()).text()

            failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
            if failed:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText(str(rtdc_ds))
                msg.setWindowTitle("Error occurred during loading file")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

            nr_images = rtdc_ds["events"]["image"].len()
            ind = np.random.randint(0,nr_images)
            img = rtdc_ds["events"]["image"][ind]
            if len(img.shape)==2:
                height, width = img.shape
                channels = 1
            elif len(img.shape)==3:
                height, width, channels = img.shape
            else:
                print("Invalid image format: "+str(img.shape))
                return


            #zoom image such that longest side is 512
            zoom_factor = np.round(float(512.0/np.max(img.shape)),0)
            #Get the order, specified in Image processing->Zoom Order
            zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order
            #Convert to corresponding cv2 zooming method
            zoom_interpol_method = aid_img.zoom_arguments_scipy2cv(zoom_factor,zoom_order)

            img_zoomed = cv2.resize(img, dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=eval(zoom_interpol_method))

            #get the location of the cell
            rowPosition = item.row()
            pix = float(self.table_dragdrop.item(rowPosition, 8).text()) # 8 PIX
            #pix = rtdc_ds.config["imaging"]["pixel size"]
            PIX = pix

            pos_x,pos_y = rtdc_ds["events"]["pos_x"][ind]/PIX,rtdc_ds["events"]["pos_y"][ind]/PIX
            cropsize = self.spinBox_imagecrop.value()
            y1 = int(round(pos_y))-cropsize/2
            x1 = int(round(pos_x))-cropsize/2
            y2 = y1+cropsize
            x2 = x1+cropsize

            #Crop the image
            img_crop = img[int(y1):int(y2),int(x1):int(x2)]
            #zoom image such that the height gets the same as for non-cropped img
            zoom_factor = float(img_zoomed.shape[0])/img_crop.shape[0]

            if zoom_factor == np.inf:
                factor = 1
# =============================================================================
#                 if self.actionVerbose.isChecked()==True:
#                     print("Set resize factor to 1. Before, it was: "+str(factor))
# =============================================================================
            #Get the order, specified in Image processing->Zoom Order
            zoom_order = str(self.comboBox_zoomOrder.currentText()) #
            zoom_interpol_method = aid_img.zoom_arguments_scipy2cv(zoom_factor,zoom_order)
            img_crop = cv2.resize(img_crop, dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=eval(zoom_interpol_method))
            #img_crop = cv2.resize(img_crop, dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=zoom_interpol_method)

            try: #add a seperate line
                grid = np.zeros((img_zoomed.shape[0],1),dtype=bool)
                sample_img = np.hstack((img_zoomed,grid,img_crop))
            except:
                grid = np.zeros((img_zoomed.shape[0],1,img_zoomed.shape[2]),dtype=bool)
                sample_img = np.hstack((img_zoomed,grid,img_crop))

            self.send_to_napari(sample_img)

    def items_clicked(self):
        #This function checks, which data has been checked on table_dragdrop and returns the necessary data
        rowCount = self.table_dragdrop.rowCount()
        #Collect urls to files that are checked
        SelectedFiles = []
        for rowPosition in range(rowCount):
            #get the filename/path
            rtdc_path = str(self.table_dragdrop.item(rowPosition, 1).text())
            #get the index (celltype) of it
            index = int(self.table_dragdrop.cellWidget(rowPosition, 2).currentIndex())
            #is it checked for train?
            cb_t = self.table_dragdrop.item(rowPosition, 3)
            #How many Events contains dataset in total?
            nr_events = int(self.table_dragdrop.item(rowPosition, 6).text())
            #how many cells/epoch during training or validation?
            nr_events_epoch = int(self.table_dragdrop.item(rowPosition, 7).text())
            #should the dataset be randomized (shuffled?)
            shuffle = bool(self.table_dragdrop.item(rowPosition, 9).checkState())
            #should the images be zoomed in/out by a factor?
            zoom_factor = float(self.table_dragdrop.item(rowPosition, 10).text())
            #should xtra_data be used for training?
            xtra_in = bool(self.table_dragdrop.item(rowPosition, 11).checkState())

            if cb_t.checkState() == QtCore.Qt.Checked and nr_events_epoch>0: #add to training files if the user wants more than 0 images per epoch
                failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
                if failed:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(str(rtdc_ds))
                    msg.setWindowTitle("Error occurred during loading file")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                hash_ = aid_bin.hashfunction(rtdc_path)#rtdc_ds.hash
                features = list(rtdc_ds["events"].keys())
                nr_images = rtdc_ds["events"]["image"].len()
                SelectedFiles.append({"rtdc_ds":rtdc_ds,"rtdc_path":rtdc_path,"features":features,"nr_images":nr_images,"class":index,"TrainOrValid":"Train","nr_events":nr_events,"nr_events_epoch":nr_events_epoch,"shuffle":shuffle,"zoom_factor":zoom_factor,"hash":hash_,"xtra_in":xtra_in})

            cb_v = self.table_dragdrop.item(rowPosition, 4)
            if cb_v.checkState() == QtCore.Qt.Checked and nr_events_epoch>0:
                failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
                if failed:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(str(rtdc_ds))
                    msg.setWindowTitle("Error occurred during loading file")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return
                hash_ = aid_bin.hashfunction(rtdc_path)
                features = list(rtdc_ds["events"].keys())
                nr_images = rtdc_ds["events"]["image"].len()
                SelectedFiles.append({"rtdc_ds":rtdc_ds,"rtdc_path":rtdc_path,"features":features,"nr_images":nr_images,"class":index,"TrainOrValid":"Valid","nr_events":nr_events,"nr_events_epoch":nr_events_epoch,"shuffle":shuffle,"zoom_factor":zoom_factor,"hash":hash_,"xtra_in":xtra_in})
        return SelectedFiles


    def send_to_napari(self,image):
        existing_layers = {layer.name for layer in self.viewer.layers}
        if "sample" in existing_layers:
            self.viewer.layers.remove("sample" )
        new_layer = self.viewer.add_image(image,name="sample")


    def uncheck_if_zero(self,item):
        #If the Nr. of epochs is changed to zero:
        #uncheck the dataset for train/valid
        row = item.row()
        col = item.column()
        #if the user changed Nr. of cells per epoch to zero
        if col==7 and int(item.text())==0:
            #get the checkstate of the coresponding T/V
            cb_t = self.table_dragdrop.item(row, 3)
            if cb_t.checkState() == QtCore.Qt.Checked:
                cb_t.setCheckState(False)
            cb_v = self.table_dragdrop.item(row, 4)
            if cb_v.checkState() == QtCore.Qt.Checked:
                cb_v.setCheckState(False)


    def select_all(self,col):
        """
        Check/Uncheck items on table_dragdrop
        """
        parent = self.sender().parent()
        if col==0:
            for i in range(parent.rowCount()):
                parent.removeRow(0)
        
        
        apply_at_col = [3,4,9,11]
        if col not in apply_at_col:
            return
        #otherwiese continue
        rows = range(self.table_dragdrop.rowCount()) #Number of rows of the table

        tableitems = [self.table_dragdrop.item(row, col) for row in rows]
        checkStates = [tableitem.checkState() for tableitem in tableitems]
        #Checked?
        checked = [state==QtCore.Qt.Checked for state in checkStates]
        if set(checked)=={True}:#all are checked!
            #Uncheck all!
            for tableitem in tableitems:
                tableitem.setCheckState(QtCore.Qt.Unchecked)
        else:#otherwise check all
            for tableitem in tableitems:
                tableitem.setCheckState(QtCore.Qt.Checked)

        #If shuffle column was clicked do some extra
        if col==9:
            for rowPosition in rows:
                if bool(self.table_dragdrop.item(rowPosition, 9).checkState())==False:
                    rtdc_path = self.table_dragdrop.item(rowPosition, 1).text()
                    rtdc_path = str(rtdc_path)

                    failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
                    if failed:
                        msg = QtWidgets.QMessageBox()
                        msg.setIcon(QtWidgets.QMessageBox.Critical)
                        msg.setText(str(rtdc_ds))
                        msg.setWindowTitle("Error occurred during loading file")
                        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        msg.exec_()
                        return
                    nr_images = rtdc_ds["events"]["image"].len()

                    columnPosition = 7
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.DisplayRole, nr_images)
                    item.setFlags(item.flags() &~QtCore.Qt.ItemIsEnabled &~ QtCore.Qt.ItemIsSelectable )
                    self.table_dragdrop.setItem(rowPosition, columnPosition, item)
                if bool(self.table_dragdrop.item(rowPosition, 9).checkState())==True:
                    #Inspect this table item. If shuffle was checked before, it will be grayed out. Invert normal cell then
                    item = self.table_dragdrop.item(rowPosition, 7)
                    item.setFlags(item.flags() |QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable )

        #Finally, update the Data-Overview-Box
        self.dataOverviewOn()#update the overview box


    def button_hist(self,item):
        buttonClicked = self.sender()
        index = self.table_dragdrop.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        rtdc_path = self.table_dragdrop.item(rowPosition, 1).text()
        rtdc_path = str(rtdc_path)

        failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
        if failed:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(str(rtdc_ds))
            msg.setWindowTitle("Error occurred during loading file")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        self.rtdc_ds = rtdc_ds
#        feature_values = rtdc_ds[feature]
        #Init a popup window
        self.w = MyPopup()
# =============================================================================
#         #set style
#         self.w.setStyleSheet(style_base)
#         self.w.setStyleSheet(style_buttons)
#         self.w.setStyleSheet(style_custom)
# =============================================================================

        self.w.setWindowTitle(rtdc_path)
        self.w.setObjectName(_fromUtf8("w"))
        self.w.resize(400, 350)
        self.gridLayout_w2 = QtWidgets.QGridLayout(self.w)
        self.gridLayout_w2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_w2.setObjectName(_fromUtf8("gridLayout_w2"))
        self.widget = QtWidgets.QWidget(self.w)
        self.widget.setMinimumSize(0, 65)
        self.widget.setMaximumSize(16777215, 65)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout_w3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_w3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_w3.setObjectName(_fromUtf8("horizontalLayout_w3"))
        self.verticalLayout_w = QtWidgets.QVBoxLayout()
        self.verticalLayout_w.setObjectName(_fromUtf8("verticalLayout_w"))
        self.horizontalLayout_w = QtWidgets.QHBoxLayout()
        self.horizontalLayout_w.setObjectName(_fromUtf8("horizontalLayout_w"))
        self.comboBox_feat1 = QtWidgets.QComboBox(self.widget)
        self.comboBox_feat1.setObjectName(_fromUtf8("comboBox_feat1"))
        features = list(self.rtdc_ds["events"].keys())
        try:
            features.remove("image")
            features.remove("mask")
        except:
            pass

        self.comboBox_feat1.addItems(features)
        self.horizontalLayout_w.addWidget(self.comboBox_feat1)
        self.comboBox_feat2 = QtWidgets.QComboBox(self.widget)
        self.comboBox_feat2.setObjectName(_fromUtf8("comboBox_feat2"))
        self.comboBox_feat2.addItems(features)
        self.horizontalLayout_w.addWidget(self.comboBox_feat2)
        self.verticalLayout_w.addLayout(self.horizontalLayout_w)
        self.horizontalLayout_w2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_w2.setObjectName(_fromUtf8("horizontalLayout_w2"))
        self.pushButton_Hist1 = QtWidgets.QPushButton(self.widget)
        self.pushButton_Hist1.setObjectName(_fromUtf8("pushButton_Hist1"))
        self.horizontalLayout_w2.addWidget(self.pushButton_Hist1)
        self.pushButton_Hist2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_Hist2.setObjectName(_fromUtf8("pushButton_Hist2"))
        self.horizontalLayout_w2.addWidget(self.pushButton_Hist2)
        self.verticalLayout_w.addLayout(self.horizontalLayout_w2)
        self.horizontalLayout_w3.addLayout(self.verticalLayout_w)
        self.verticalLayout_w2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_w2.setObjectName(_fromUtf8("verticalLayout_w2"))
        self.pushButton_Scatter = QtWidgets.QPushButton(self.widget)
        self.pushButton_Scatter.setObjectName(_fromUtf8("pushButton_Scatter"))
        self.verticalLayout_w2.addWidget(self.pushButton_Scatter)
        self.checkBox_ScalePix = QtWidgets.QCheckBox(self.widget)
        self.checkBox_ScalePix.setObjectName(_fromUtf8("checkBox_ScalePix"))
        self.verticalLayout_w2.addWidget(self.checkBox_ScalePix)
        self.horizontalLayout_w3.addLayout(self.verticalLayout_w2)
        self.gridLayout_w2.addWidget(self.widget, 0, 0, 1, 1)


        self.pushButton_Hist1.setText("Hist")
        self.pushButton_Hist1.clicked.connect(self.update_hist1)
        self.pushButton_Hist2.setText("Hist")
        self.pushButton_Hist2.clicked.connect(self.update_hist2)
        self.pushButton_Scatter.setText("Scatter")
        self.pushButton_Scatter.clicked.connect(self.update_scatter)

        self.checkBox_ScalePix.setText("Scale by pix")

        self.histogram = pg.GraphicsWindow()
        self.plt1 = self.histogram.addPlot()
#        y,x = np.histogram(feature_values, bins='auto')
#        plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        self.gridLayout_w2.addWidget(self.histogram,1, 0, 1, 1)
        self.w.show()

    def update_hist1(self):
        feature = str(self.comboBox_feat1.currentText())
        feature_values = self.rtdc_ds["events"][feature]
        y,x = np.histogram(feature_values, bins='auto')
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),clear=True)

    def update_hist2(self):
        feature = str(self.comboBox_feat2.currentText())
        feature_values = self.rtdc_ds["events"][feature]
        y,x = np.histogram(feature_values, bins='auto')
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),clear=True)

    def update_scatter(self):
        feature_x = str(self.comboBox_feat1.currentText())
        feature_x_values = self.rtdc_ds["events"][feature_x]
        feature_y = str(self.comboBox_feat2.currentText())
        feature_y_values = self.rtdc_ds["events"][feature_y]
        if len(feature_x_values)==len(feature_y_values):
            self.plt1.plot(feature_x_values, feature_y_values,pen=None,symbol='o',clear=True)

    def action_show_example_imgs(self):
        #Get state of the comboboxes!
        tr_or_valid = str(self.comboBox_example_train.currentText())
        w_or_wo_augm = str(self.comboBox_example_aug.currentText())

        #most of it should be similar to action_fit_model_worker
        #Used files go to a separate sheet on the MetaFile.xlsx
        SelectedFiles = self.items_clicked_no_rtdc_ds()
        #Collect all information about the fitting routine that was user defined
        crop = int(self.spinBox_imagecrop.value())
        norm = str(self.comboBox_Normalization.currentText())
        h_flip = bool(self.checkBox_HorizFlip.isChecked())
        v_flip = bool(self.checkBox_VertFlip.isChecked())

        rotation = float(self.lineEdit_Rotation.text())
        width_shift = float(self.lineEdit_widthShift.text())
        height_shift = float(self.lineEdit_heightShift.text())
        zoom = float(self.lineEdit_zoomRange.text())
        shear = float(self.lineEdit_shearRange.text())

        brightness_add_lower = float(self.spinBox_PlusLower.value())
        brightness_add_upper = float(self.spinBox_PlusUpper.value())
        brightness_mult_lower = float(self.doubleSpinBox_MultLower.value())
        brightness_mult_upper = float(self.doubleSpinBox_MultUpper.value())
        gaussnoise_mean = float(self.doubleSpinBox_GaussianNoiseMean.value())
        gaussnoise_scale = float(self.doubleSpinBox_GaussianNoiseScale.value())

        contrast_on = bool(self.checkBox_contrast.isChecked())
        contrast_lower = float(self.doubleSpinBox_contrastLower.value())
        contrast_upper = float(self.doubleSpinBox_contrastHigher.value())
        saturation_on = bool(self.checkBox_saturation.isChecked())
        saturation_lower = float(self.doubleSpinBox_saturationLower.value())
        saturation_upper = float(self.doubleSpinBox_saturationHigher.value())
        hue_on = bool(self.checkBox_hue.isChecked())
        hue_delta = float(self.doubleSpinBox_hueDelta.value())

        avgBlur_on = bool(self.checkBox_avgBlur.isChecked())
        avgBlur_min = int(self.spinBox_avgBlurMin.value())
        avgBlur_max = int(self.spinBox_avgBlurMax.value())

        gaussBlur_on = bool(self.checkBox_gaussBlur.isChecked())
        gaussBlur_min = int(self.spinBox_gaussBlurMin.value())
        gaussBlur_max = int(self.spinBox_gaussBlurMax.value())

        motionBlur_on = bool(self.checkBox_motionBlur.isChecked())
        motionBlur_kernel = str(self.lineEdit_motionBlurKernel.text())
        motionBlur_angle = str(self.lineEdit_motionBlurAngle.text())

        motionBlur_kernel = tuple(ast.literal_eval(motionBlur_kernel)) #translate string in the lineEdits to a tuple
        motionBlur_angle = tuple(ast.literal_eval(motionBlur_angle)) #translate string in the lineEdits to a tuple

        paddingMode = str(self.comboBox_paddingMode.currentText())#.lower()

        #which index is requested by user:?
        req_index = int(self.comboBox_ShowIndex.currentText())


        if tr_or_valid=='Training':
            ######################Load the Training Data################################
            ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
        elif tr_or_valid=='Validation':
            ind = [selectedfile["TrainOrValid"] == "Valid" for selectedfile in SelectedFiles]
        ind = np.where(np.array(ind)==True)[0]
        SelectedFiles = np.array(SelectedFiles)[ind]
        SelectedFiles = list(SelectedFiles)
        indices = [selectedfile["class"] for selectedfile in SelectedFiles]
        ind = np.where(np.array(indices)==req_index)[0]
        if len(ind)<1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("There is no data for this class available")
            msg.setWindowTitle("Class not available")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        indices = list(np.array(indices)[ind])
        SelectedFiles = list(np.array(SelectedFiles)[ind])
        nr_events_epoch = len(indices)*[10] #[selectedfile["nr_events_epoch"] for selectedfile in SelectedFiles]
        rtdc_path = [selectedfile["rtdc_path"] for selectedfile in SelectedFiles]
        zoom_factors = [selectedfile["zoom_factor"] for selectedfile in SelectedFiles]
        zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order
        shuffle = [selectedfile["shuffle"] for selectedfile in SelectedFiles]
        #If the scaling method is "divide by mean and std of the whole training set":
        if norm == "StdScaling using mean and std of all training data":
            mean_trainingdata,std_trainingdata = [],[]
            for i in range(len(SelectedFiles)):
                gen = aid_img.gen_crop_img(crop,rtdc_path[i],random_images=False,zoom_factor=zoom_factors[i],zoom_order=zoom_order,color_mode=self.comboBox_GrayOrRGB.currentText(),padding_mode=paddingMode)

                images = next(gen)[0]
                mean_trainingdata.append(np.mean(images))
                std_trainingdata.append(np.std(images))
            mean_trainingdata = np.mean(np.array(mean_trainingdata))
            std_trainingdata = np.mean(np.array(std_trainingdata))
            if np.allclose(std_trainingdata,0):
                std_trainingdata = 0.0001
                print("std_trainingdata was zero and is now set to 0.0001 to avoid div. by zero!")

        if w_or_wo_augm=='Augm.':

            ###############Continue with training data:augmentation############
            #Rotating could create edge effects. Avoid this by making crop a bit larger for now
            #Worst case would be a 45degree rotation:
            cropsize2 = np.sqrt(crop**2+crop**2)
            cropsize2 = np.ceil(cropsize2 / 2.) * 2 #round to the next even number

            ############Cropping and image augmentation#####################
            #Start the first iteration:
            X,y = [],[]
            for i in range(len(SelectedFiles)):
                gen = aid_img.gen_crop_img(cropsize2,rtdc_path[i],10,random_images=True,replace=True,zoom_factor=zoom_factors[i],\
                                           zoom_order=zoom_order,color_mode=self.comboBox_GrayOrRGB.currentText(),padding_mode=paddingMode)
                print("gen success")

                try: #When all cells are at the border of the image, the generator will be empty. Avoid program crash by try, except
                    X.append(next(gen)[0])
                except StopIteration:
                    print("All events at border of image and discarded")
                    return
                y.append(np.repeat(indices[i],X[-1].shape[0]))

            X = np.concatenate(X)
            X = X.astype(np.uint8) #make sure we stay in uint8
            y = np.concatenate(y)

            if len(X.shape)==4:
                channels=3
            elif len(X.shape)==3:
                channels=1
                X = np.expand_dims(X,3)#Add the "channels" dimension
            else:
                print("Invalid data dimension:" +str(X.shape))

            X_batch, y_batch = aid_img.affine_augm(X,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear), y #Affine image augmentation
            X_batch = X_batch.astype(np.uint8) #make sure we stay in uint8

            #Now do the final cropping to the actual size that was set by user
            dim = X_batch.shape
            if dim[2]!=crop:
                remove = int(dim[2]/2.0 - crop/2.0)
                #X_batch = X_batch[:,:,remove:-remove,remove:-remove] #crop to crop x crop pixels #Theano
                X_batch = X_batch[:,remove:remove+crop,remove:remove+crop,:] #crop to crop x crop pixels #TensorFlow

            ##########Contrast/Saturation/Hue augmentation#########
            #is there any of contrast/saturation/hue augmentation to do?
            if contrast_on:
                X_batch = aid_img.contrast_augm_cv2(X_batch,contrast_lower,contrast_upper) #this function is almost 15 times faster than random_contrast from tf!
            if saturation_on or hue_on:
                X_batch = aid_img.satur_hue_augm_cv2(X_batch.astype(np.uint8),saturation_on,saturation_lower,saturation_upper,hue_on,hue_delta)

            ##########Average/Gauss/Motion blurring#########
            #is there any of blurring to do?
            if avgBlur_on:
                X_batch = aid_img.avg_blur_cv2(X_batch,avgBlur_min,avgBlur_max)
            if gaussBlur_on:
                X_batch = aid_img.gauss_blur_cv(X_batch,gaussBlur_min,gaussBlur_max)
            if motionBlur_on:
                X_batch = aid_img.motion_blur_cv(X_batch,motionBlur_kernel,motionBlur_angle)

            X_batch = aid_img.brightn_noise_augm_cv2(X_batch,brightness_add_lower,brightness_add_upper,brightness_mult_lower,brightness_mult_upper,gaussnoise_mean,gaussnoise_scale)

            if norm == "StdScaling using mean and std of all training data":
                X_batch = aid_img.image_normalization(X_batch,norm,mean_trainingdata,std_trainingdata)
            else:
                X_batch = aid_img.image_normalization(X_batch,norm)

            X = X_batch
            #if verbose: print("Shape of the shown images is:"+str(X.shape))

        elif w_or_wo_augm=='Original':
            ############Cropping#####################
            X,y = [],[]
            for i in range(len(SelectedFiles)):
                gen = aid_img.gen_crop_img(crop,rtdc_path[i],10,random_images=True,replace=True,zoom_factor=zoom_factors[i],zoom_order=zoom_order,color_mode=self.comboBox_GrayOrRGB.currentText(),padding_mode=paddingMode)

                try:
                    X.append(next(gen)[0])

                except:
                    return
                y.append(np.repeat(indices[i],X[-1].shape[0]))

            X = np.concatenate(X)
            y = np.concatenate(y)

            if len(X.shape)==4:
                channels=3
            elif len(X.shape)==3:
                channels=1
                X = np.expand_dims(X,3) #Add the "channels" dimension
            else:
                print("Invalid data dimension: " +str(X.shape))

        if norm == "StdScaling using mean and std of all training data":
            X = aid_img.image_normalization(X,norm,mean_trainingdata,std_trainingdata)
        else:
            X = aid_img.image_normalization(X,norm)



        imgs = []
        for i in range(9):
            if channels==1:
                img = X[i,:,:,0] #TensorFlow
            if channels==3:
                img = X[i,:,:,:] #TensorFlow


            #Stretch pixel value to full 8bit range (0-255); only for display
            img = img-np.min(img)
            fac = np.max(img)
            img = (img/fac)*255.0
            img = img.astype(np.uint8)
            imgs.append(img)

        rows = [np.hstack((imgs[i*3],imgs[i*3+1],imgs[i*3+2])) for i in range(3)]
        img = np.vstack(rows)

        self.send_to_napari(img)


# =============================================================================
#             if channels==1:
#                 height, width = img.shape
#             if channels==3:
#                 height, width, _ = img.shape
#
#             if channels==1:
#                 self.send_to_napari(img)
#                 #self.image_show.setImage(img.T,autoRange=False)
#             if channels==3:
#                 self.send_to_napari(img)
#                 #self.image_show.setImage(np.swapaxes(img,0,1),autoRange=False)
# =============================================================================

    def get_color_mode(self):
        if str(self.comboBox_GrayOrRGB.currentText())=="Grayscale":
            return "Grayscale"
        elif str(self.comboBox_GrayOrRGB.currentText())=="RGB":
            return "RGB"
        else:
            return None

    def get_metrics(self):
        Metrics =  []
# =============================================================================
#         #no expert mode
#         f1 = bool(self.checkBox_expertF1.isChecked())
#         if f1==True:
#             Metrics.append("auc")
#         precision = bool(self.checkBox_expertPrecision.isChecked())
#         if precision==True:
#             Metrics.append("precision")
#         recall = bool(self.checkBox_expertRecall.isChecked())
#         if recall==True:
#             Metrics.append("recall")
# =============================================================================
        metrics =  ['accuracy'] + Metrics
        #metrics = aid_dl.get_metrics_tensors(metrics,nr_classes)
        return metrics


    def action_preview_model(self,enabled):#function runs when radioButton_LoadRestartModel or radioButton_LoadContinueModel was clicked
        if enabled:
            #if the "Load and restart" radiobutton was clicked:
            if self.radioButton_LoadRestartModel.isChecked():
                modelname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open model architecture', Default_dict["Path of last model"],"Architecture or model (*.arch *.model)")
                modelname = modelname[0]
                #modelname_for_dict = modelname
            #if the "Load and continue" radiobutton was clicked:
            elif self.radioButton_LoadContinueModel.isChecked():
                modelname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open model with all parameters', Default_dict["Path of last model"],"Keras model (*.model)")
                modelname = modelname[0]
                #modelname_for_dict = modelname
            self.lineEdit_LoadModelPath.setText(modelname) #Put the filename to the line edit

            #Remember the location for next time
            if len(str(modelname))>0:
                Default_dict["Path of last model"] = os.path.split(modelname)[0]
                aid_bin.save_aid_settings(Default_dict)
            #If user wants to load and restart a model
            if self.radioButton_LoadRestartModel.isChecked():
                #load the model and print summary
                if modelname.endswith(".arch"):
                    json_file = open(modelname)
                    model_config = json_file.read()
                    json_file.close()
                    model_config = json.loads(model_config)
                    #cut the .json off
                    modelname = modelname.split(".arch")[0]

                #Or a .model (FULL model with trained weights) , but for display only load the architecture
                elif modelname.endswith(".model"):
                    #Load the model config (this is the architecture)
                    model_full_h5 = h5py.File(modelname, 'r')
                    model_config = model_full_h5.attrs['model_config']
                    model_full_h5.close() #close the hdf5
                    model_config = json.loads(model_config)
                    #model = model_from_config(model_config)
                    modelname = modelname.split(".model")[0]
                else:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No valid file was chosen. Please specify a file that was created using AIDeveloper or Keras")
                    msg.setWindowTitle("No valid file was chosen")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return
                    #raise ValueError("No valid file was chosen")

                text1 = "Architecture: loaded from .arch\nWeights: will be randomly initialized'\n"

                #Try to find the corresponding .meta
                #All models have a number:
                metaname = modelname.rsplit('_',1)[0]+"_meta.xlsx"
                if os.path.isfile(metaname):
                    #open the metafile
                    meta = pd.read_excel(metaname,sheet_name="Parameters",engine="openpyxl")
                    if "Chosen Model" in list(meta.keys()):
                        chosen_model = meta["Chosen Model"].iloc[-1]
                    else:
                        #Try to get the model architecture and adjust the combobox
                        try:
                            ismlp,chosen_model = model_zoo.mlpconfig_to_str(model_config)
                        except:#No model could be identified
                            chosen_model = "None"
                else:
                    #Try to get the model architecture and adjust the combobox
                    try:
                        ismlp,chosen_model = model_zoo.mlpconfig_to_str(model_config)
                    except:#No model could be identified
                        chosen_model = "None"

                if chosen_model is not None:
                    #chosen_model is a string that should be contained in comboBox_ModelSelection
                    index = self.comboBox_ModelSelection.findText(chosen_model, QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.comboBox_ModelSelection.setCurrentIndex(index)
                else:
                    index = self.comboBox_ModelSelection.findText('None', QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.comboBox_ModelSelection.setCurrentIndex(index)


            #Otherwise, user wants to load and continue training a model
            elif self.radioButton_LoadContinueModel.isChecked():
                #User can only choose a .model (FULL model with trained weights) , but for display only load the architecture
                if modelname.endswith(".model"):
                    #Load the model config (this is the architecture)
                    model_full_h5 = h5py.File(modelname, 'r')
                    model_config = model_full_h5.attrs['model_config']
                    model_full_h5.close() #close the hdf5
                    model_config = json.loads(model_config)
                    #model = model_from_config(model_config)
                    modelname = modelname.split(".model")[0]

                    #Try to find the corresponding .meta
                    #All models have a number:
                    metaname = modelname.rsplit('_',1)[0]+"_meta.xlsx"
                    if os.path.isfile(metaname):
                        #open the metafile
                        meta = pd.read_excel(metaname,sheet_name="Parameters",engine="openpyxl")
                        if "Chosen Model" in list(meta.keys()):
                            chosen_model = meta["Chosen Model"].iloc[-1]
                        else:
                            #Try to get the model architecture and adjust the combobox
                            try:
                                ismlp,chosen_model = model_zoo.mlpconfig_to_str(model_config)
                            except:#No model could be identified
                                chosen_model = "None"
                    else:
                        #Try to get the model architecture and adjust the combobox
                        try:
                            ismlp,chosen_model = model_zoo.mlpconfig_to_str(model_config)
                        except:#No model could be identified
                            chosen_model = "None"

                    if chosen_model is not None:
                        #chosen_model is a string that should be contained in comboBox_ModelSelection
                        index = self.comboBox_ModelSelection.findText(chosen_model, QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_ModelSelection.setCurrentIndex(index)
                    else:
                        index = self.comboBox_ModelSelection.findText('None', QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_ModelSelection.setCurrentIndex(index)
                    text1 = "Architecture: loaded from .model\nWeights: pretrained weights will be loaded and used when hitting button 'Initialize model!'\n"
                else:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No valid file was chosen. Please specify a file that was created using AIDeveloper or Keras")
                    msg.setWindowTitle("No valid file was chosen")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return
                    #raise ValueError("No valid file was chosen")

            #In both cases (restart or continue) the input dimensions have to fit
            #The number of output classes should also fit but this is not essential
            #but most users certainly want the same number of classes (output)->Give Info
            in_dim, out_dim = aid_dl.model_in_out_dim(model_config,"config")

            #Retrieve the color_mode from the model (nr. of channels in last in_dim)
            channels = in_dim[-1] #TensorFlow: channels in last dimension
            if channels==1:
                channel_text = "1 channel (Grayscale)"
                if self.get_color_mode()!="Grayscale":
                    #when model needs Grayscale, set the color mode in comboBox_GrayOrRGB to that
                    index = self.comboBox_GrayOrRGB.findText("Grayscale", QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.comboBox_GrayOrRGB.setCurrentIndex(index)
# =============================================================================
#                     self.statusbar.showMessage("Color Mode set to Grayscale",5000)
# =============================================================================

            elif channels==3:
                channel_text = "3 channels (RGB)"
                if self.get_color_mode()!="RGB":
                    #when model needs RGB, set the color mode in the ui to that
                    index = self.comboBox_GrayOrRGB.findText("RGB", QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.comboBox_GrayOrRGB.setCurrentIndex(index)
# =============================================================================
#                     self.statusbar.showMessage("Color Mode set to RGB",5000)
# =============================================================================

            text2 = "Model Input: loaded Model takes: "+str(in_dim[-3])+" x "+str(in_dim[-2]) + " pixel images and "+channel_text+"\n"
            if int(self.spinBox_imagecrop.value())!=int(in_dim[-2]):
                self.spinBox_imagecrop.setValue(in_dim[-2])
                text2 = text2+ "'Input image size'  in GUI was changed accordingly\n"

            #check that the nr. of classes are equal to the model out put
            SelectedFiles = self.items_clicked_no_rtdc_ds()
            indices = [s["class"] for s in SelectedFiles]

            nr_classes = np.max(indices)+1

            if int(nr_classes)==int(out_dim):
                text3 = "Output: "+str(out_dim)+" classes\n"
            elif int(nr_classes)>int(out_dim):#Dataset has more classes than the model provides!
                text3 = "Loaded model has only "+(str(out_dim))+\
                " output nodes (classes) but your selected data has "+str(nr_classes)+\
                " classes. Therefore, the model will be adjusted before fitting, by customizing the final Dense layer.\n"
                #aid_dl.model_add_classes(model_keras,nr_classes)#this function changes model_keras inplace
            elif int(nr_classes)<int(out_dim):#Dataset has less classes than the model provides!
                text3 = "Model output: The architecture you chose has "+(str(out_dim))+\
                " output nodes (classes) and your selected data has only "+str(nr_classes)+\
                " classes. This is fine. The model will essentially have some excess classes that are not used.\n"

            text = text1+text2+text3
            self.textBrowser_Info.setText(text)

            if self.radioButton_LoadContinueModel.isChecked():
                #"Load the parameter file of the model that should be continued and apply the same normalization"
                #Make a popup: You are about to continue to train a pretrained model
                #Please select the parameter file of that model to load the normalization method
                #or choose the normalization method manually:
                #this is important
                self.popup_normalization()

    def action_set_modelpath_and_name(self):
        #Get the path and filename for the new model
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save model', Default_dict["Path of last model"],"Keras Model file (*.model)")
        filename = filename[0]
        if len(filename)==0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("No valid filename was chosen.")
            msg.setWindowTitle("No valid filename was chosen")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        if filename.endswith(".arch"):
            filename = filename.split(".arch")[0]
        #add the suffix .model
        if not filename.endswith(".model"):
            filename = filename +".model"
        self.lineEdit_modelname.setText(filename)
        #Write to Default_dict
        Default_dict["Path of last model"] = os.path.split(filename)[0]
        aid_bin.save_aid_settings(Default_dict)


    def action_initialize_model(self,duties="initialize_train"):
        """
        duties: which tasks should be performed: "initialize", "initialize_train", "initialize_lrfind"
        """
        #print("duties: "+str(duties))

        #Create config (define which device to use)
        if self.radioButton_cpu.isChecked():
            deviceSelected = str(self.comboBox_cpu.currentText())
        elif self.radioButton_gpu.isChecked():
            deviceSelected = str(self.comboBox_gpu.currentText())
        gpu_memory = float(self.doubleSpinBox_memory.value())
        config_gpu = aid_dl.get_config(cpu_nr,gpu_nr,deviceSelected,gpu_memory)

#        try:
#            K.clear_session()
#        except:
#            print("Could not clear_session (7)")

        with tf.compat.v1.Session(graph = tf.Graph(), config=config_gpu) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            #Initialize the model
            ###########################New model###################################
            if self.radioButton_NewModel.isChecked():
                load_modelname = "" #No model is loaded
                text0 = load_modelname
                #Create a new model!
                #Get what the user wants from the dropdown menu!
                chosen_model = str(self.comboBox_ModelSelection.currentText())
                if chosen_model==None:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No model specified!")
                    msg.setWindowTitle("No model specified!")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                in_dim = int(self.spinBox_imagecrop.value())
                SelectedFiles = self.items_clicked()
                #rtdc_ds = SelectedFiles[0]["rtdc_ds"]

                if str(self.comboBox_GrayOrRGB.currentText())=="Grayscale":
                    channels=1
                elif str(self.comboBox_GrayOrRGB.currentText())=="RGB":
                    channels=3

                indices = [s["class"] for s in SelectedFiles]
                indices_unique = np.unique(np.array(indices))
                if len(indices_unique)<2:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("Need at least two classes to fit. Please specify .rtdc files and corresponding indeces")
                    msg.setWindowTitle("No valid file was chosen")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                out_dim = np.max(indices)+1
                nr_classes = out_dim

                if chosen_model=="None":
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No model specified!")
                    msg.setWindowTitle("No model specified!")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return



                try:
                    model_keras = model_zoo.get_model(chosen_model,in_dim,channels,out_dim)
                    print("model_keras",model_keras)
                    print("model_keras.optimizer",model_keras.optimizer)
# =============================================================================
#                     print("chosen_model:",chosen_model)
#                     print("in_dim:",in_dim)
#                     print("channels:",channels)
#                     print("out_dim:",out_dim)
# =============================================================================
                except Exception as e:
                    #There is an issue building the model!
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText(str(e))
                    msg.setWindowTitle("Error occured when building model:")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                text1 = "Architecture: created "+chosen_model+" design\nWeights: Initialized random weights\n"

                if self.get_color_mode()=="Grayscale":
                    channels = 1
                    channel_text = "1 channel (Grayscale)"
                elif self.get_color_mode()=="RGB":
                    channels = 3
                    channel_text = "3 channels (RGB)"

                text2 = "Model Input: "+str(in_dim)+" x "+str(in_dim) + " pixel images and "+channel_text+"\n"

                if int(nr_classes)==int(out_dim):
                    text3 = "Output: "+str(out_dim)+" classes\n"
                elif int(nr_classes)>int(out_dim):#Dataset has more classes than the model provides!
                    text3 = "Loaded model has only "+(str(out_dim))+\
                    " output nodes (classes) but your selected data has "+str(nr_classes)+\
                    " classes. Therefore, the model will be adjusted before fitting, by customizing the final Dense layer.\n"
                    aid_dl.model_add_classes(model_keras,nr_classes)#this function changes model_keras inplace
                elif int(nr_classes)<int(out_dim):#Dataset has less classes than the model provides!
                    text3 = "Model output: The architecture you chose has "+(str(out_dim))+\
                    " output nodes (classes) and your selected data has only "+str(nr_classes)+\
                    " classes. This is fine. The model will essentially have some excess classes that are not used.\n"

            #######################Load and restart model##########################
            elif self.radioButton_LoadRestartModel.isChecked():

                load_modelname = str(self.lineEdit_LoadModelPath.text())
                text0 = "Loaded model: "+load_modelname
                #load the model and print summary
                if load_modelname.endswith(".arch"):
                    json_file = open(load_modelname)
                    model_config = json_file.read()
                    json_file.close()
                    model_keras = model_from_json(model_config)
                    model_config = json.loads(model_config)
                    text1 = "\nArchitecture: loaded from .arch\nWeights: randomly initialized\n"

                #Or a .model (FULL model with trained weights) , but for display only load the architecture
                elif load_modelname.endswith(".model"):
                    #Load the model config (this is the architecture)
                    model_full_h5 = h5py.File(load_modelname, 'r')
                    model_config = model_full_h5.attrs['model_config']
                    model_full_h5.close() #close the hdf5
                    model_config = json.loads(model_config)
                    model_keras = model_from_config(model_config)
                    text1 = "\nArchitecture: loaded from .model\nWeights: randomly initialized\n"
                else:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No valid file was chosen. Please specify a file that was created using AIDeveloper or Keras")
                    msg.setWindowTitle("No valid file was chosen")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return

                try:
                    metaname = load_modelname.rsplit('_',1)[0]+"_meta.xlsx"
                    if os.path.isfile(metaname):
                        #open the metafile
                        meta = pd.read_excel(metaname,sheet_name="Parameters",engine="openpyxl")
                        if "Chosen Model" in list(meta.keys()):
                            chosen_model = meta["Chosen Model"].iloc[-1]
                except:
                    chosen_model = str(self.comboBox_ModelSelection.currentText())

                #In both cases (restart or continue) the input dimensions have to fit
                #The number of output classes should also fit but this is not essential
                #but most users certainly want the same number of classes (output)->Give Info

                in_dim, out_dim = aid_dl.model_in_out_dim(model_config,"config")

                channels = in_dim[-1] #TensorFlow: channels in last dimension

                #Compile model (consider user-specific metrics)
                model_metrics = self.get_metrics()

                model_keras.compile(loss='categorical_crossentropy',optimizer='adam',metrics=aid_dl.get_metrics_tensors(model_metrics,out_dim))#dont specify loss and optimizer yet...expert stuff will follow and model will be recompiled

                if channels==1:
                    channel_text = "1 channel (Grayscale)"
                    if self.get_color_mode()!="Grayscale":
                        #when model needs Grayscale, set the color mode in comboBox_GrayOrRGB to that
                        index = self.comboBox_GrayOrRGB.findText("Grayscale", QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_GrayOrRGB.setCurrentIndex(index)
                        self.statusbar.showMessage("Color Mode set to Grayscale",5000)

                elif channels==3:
                    channel_text = "3 channels (RGB)"
                    if self.get_color_mode()!="RGB":
                        #when model needs RGB, set the color mode in the ui to that
                        index = self.comboBox_GrayOrRGB.findText("RGB", QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_GrayOrRGB.setCurrentIndex(index)
                        self.statusbar.showMessage("Color Mode set to RGB",5000)

                text2 = "Model Input: "+str(in_dim[-3])+" x "+str(in_dim[-2]) + " pixel images and "+channel_text+"\n"

                if int(self.spinBox_imagecrop.value())!=int(in_dim[-2]):
                    self.spinBox_imagecrop.setValue(in_dim[-2])
                    text2 = text2+ "'Input image size'  in GUI was changed accordingly\n"

                #check that the nr. of classes are equal to the model out put
                SelectedFiles = self.items_clicked()
                indices = [s["class"] for s in SelectedFiles]
                nr_classes = np.max(indices)+1

                if int(nr_classes)==int(out_dim):
                    text3 = "Output: "+str(out_dim)+" classes\n"
                elif int(nr_classes)>int(out_dim):#Dataset has more classes than the model provides!
                    text3 = "Loaded model has only "+(str(out_dim))+\
                    " output nodes (classes) but your selected data has "+str(nr_classes)+\
                    " classes. Therefore, the model will be adjusted before fitting, by customizing the final Dense layer.\n"
                    aid_dl.model_add_classes(model_keras,nr_classes)#this function changes model_keras inplace
                elif int(nr_classes)<int(out_dim):#Dataset has less classes than the model provides!
                    text3 = "Model output: The architecture you chose has "+(str(out_dim))+\
                    " output nodes (classes) and your selected data has only "+str(nr_classes)+\
                    " classes. This is fine. The model will essentially have some excess classes that are not used.\n"

            ###############Load and continue training the model####################
            elif self.radioButton_LoadContinueModel.isChecked():
                load_modelname = str(self.lineEdit_LoadModelPath.text())
                text0 = "Loaded model: "+load_modelname+"\n"

                #User can only choose a .model (FULL model with trained weights) , but for display only load the architecture
                if load_modelname.endswith(".model"):
                    #Load the full model
                    try:
                        model_keras = load_model(load_modelname,custom_objects=aid_dl.get_custom_metrics())
                    except:
                        K.clear_session() #On linux It happened that there was an error, if another fitting run before
                        model_keras = load_model(load_modelname,custom_objects=aid_dl.get_custom_metrics())
                    #model_config = model_keras.config() #Load the model config (this is the architecture)
                    #load_modelname = load_modelname.split(".model")[0]
                    text1 = "Architecture: loaded from .model\nWeights: pretrained weights were loaded\n"
                else:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("No valid file was chosen. Please specify a file that was created using AIDeveloper or Keras")
                    msg.setWindowTitle("No valid file was chosen")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()
                    return
                    #raise ValueError("No valid file was chosen")

                try:
                    metaname = load_modelname.rsplit('_',1)[0]+"_meta.xlsx"
                    if os.path.isfile(metaname):
                        #open the metafile
                        meta = pd.read_excel(metaname,sheet_name="Parameters",engine="openpyxl")
                        if "Chosen Model" in list(meta.keys()):
                            chosen_model = meta["Chosen Model"].iloc[-1]
                    else:
                        chosen_model = str(self.comboBox_ModelSelection.currentText())

                except:
                    chosen_model = str(self.comboBox_ModelSelection.currentText())


                #Check input dimensions
                #The number of output classes should also fit but this is not essential
                #but most users certainly want the same number of classes (output)->Give Info
    #            in_dim = model_config['config'][0]['config']['batch_input_shape']
    #            out_dim = model_config['config'][-2]['config']['units']
                in_dim = model_keras.get_input_shape_at(0)
                out_dim = model_keras.get_output_shape_at(0)[1]
                channels = in_dim[-1] #TensorFlow: channels in last dimension

                if channels==1:
                    channel_text = "1 channel (Grayscale)"
                    if self.get_color_mode()!="Grayscale":
                        #when model needs Grayscale, set the color mode in comboBox_GrayOrRGB to that
                        index = self.comboBox_GrayOrRGB.findText("Grayscale", QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_GrayOrRGB.setCurrentIndex(index)
                        self.statusbar.showMessage("Color Mode set to Grayscale",5000)

                elif channels==3:
                    channel_text = "3 channels (RGB)"
                    if self.get_color_mode()!="RGB":
                        #when model needs RGB, set the color mode in the ui to that
                        index = self.comboBox_GrayOrRGB.findText("RGB", QtCore.Qt.MatchFixedString)
                        if index >= 0:
                            self.comboBox_GrayOrRGB.setCurrentIndex(index)
                        self.statusbar.showMessage("Color Mode set to RGB",5000)

                text2 = "Model Input: "+str(in_dim[-3])+" x "+str(in_dim[-2]) + " pixel images and "+channel_text+"\n"
                if int(self.spinBox_imagecrop.value())!=int(in_dim[-2]):
                    self.spinBox_imagecrop.setValue(in_dim[-2])
                    text2 = text2+ "'Input image size'  in GUI was changed accordingly\n"

                #check that the nr. of classes are equal to the model out put
                SelectedFiles = self.items_clicked()
                indices = [s["class"] for s in SelectedFiles]
                nr_classes = np.max(indices)+1

                if int(nr_classes)==int(out_dim):
                    text3 = "Output: "+str(out_dim)+" classes\n"
                elif int(nr_classes)>int(out_dim):#Dataset has more classes than the model provides!
                    text3 = "Loaded model has only "+(str(out_dim))+\
                    " output nodes (classes) but your selected data has "+str(nr_classes)+\
                    " classes. Therefore, the model will be adjusted before fitting, by customizing the final Dense layer.\n"
                    aid_dl.model_add_classes(model_keras,nr_classes)#this function changes model_keras inplace
                elif int(nr_classes)<int(out_dim):#Dataset has less classes than the model provides!
                    text3 = "Model output: The architecture you chose has "+(str(out_dim))+\
                    " output nodes (classes) and your selected data has only "+str(nr_classes)+\
                    " classes. This is fine. The model will essentially have some excess classes that are not used.\n"


            else:
                #No radio-button was chosen
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText("Please use the radiobuttons to define the model")
                msg.setWindowTitle("No model defined")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
                return


# =============================================================================
#             #If expert mode is on, apply the requested options
#             #This affects learning rate, trainability of layers and dropout rate
#             expert_mode = bool(self.groupBox_expertMode.isChecked())
#             learning_rate_const = float(self.doubleSpinBox_learningRate.value())
#             learning_rate_expert_on = bool(self.groupBox_learningRate.isChecked())
#             train_last_layers = bool(self.checkBox_trainLastNOnly.isChecked())
#             train_last_layers_n = int(self.spinBox_trainLastNOnly.value())
#             train_dense_layers = bool(self.checkBox_trainDenseOnly.isChecked())
#             dropout_expert_on = bool(self.checkBox_dropout.isChecked())
#             loss_expert_on = bool(self.checkBox_expt_loss.isChecked())
#             loss_expert = str(self.comboBox_expt_loss.currentText()).lower()
#             optimizer_expert_on = bool(self.checkBox_optimizer.isChecked())
# =============================================================================
            learning_rate_const = 0.001
            loss_expert = "categorical_crossentropy"
            optimizer_expert = "Adam"


            optimizer_settings = self.optimizer_settings.copy() #get the current optimizer settings
# =============================================================================
#             paddingMode = str(self.comboBox_paddingMode.currentText())#.lower()
#             model_metrics = self.get_metrics()
# =============================================================================

# =============================================================================
#             ?why use loss_expert?
# =============================================================================
            if "collection" in chosen_model.lower():
                for m in model_keras[1]: #in a collection, model_keras[0] are the names of the models and model_keras[1] is a list of all models
                    model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                    aid_dl.model_compile(m,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)


            if not "collection" in chosen_model.lower():
                model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                aid_dl.model_compile(model_keras,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)


# =============================================================================
#             try:
#                 dropout_expert = str(self.lineEdit_dropout.text()) #due to the validator, there are no squ.brackets
#                 dropout_expert = "["+dropout_expert+"]"
#                 dropout_expert = ast.literal_eval(dropout_expert)
#             except:
#                 dropout_expert = []
# =============================================================================

            if type(model_keras)==tuple:#when user chose a Collection of models, a tuple is returned by model_zoo.get_model
                collection = True
            else:
                collection = False

            if collection==False: #if there is a single model:
                #Original learning rate (before expert mode is switched on!)
                try:
                    self.learning_rate_original = model_keras.optimizer.get_config()["learning_rate"]
                except:
                    print("Session busy. Try again in fresh session...")
                    #tf.reset_default_graph() #Make sure to start with a fresh session
                    K.clear_session()
                    sess = tf.compat.v1.Session(graph = tf.Graph(), config=config_gpu)
                    #K.set_session(sess)
                    self.learning_rate_original = model_keras.optimizer.get_config()["learning_rate"]

                #Get initial trainability states of model
                self.trainable_original, self.layer_names = aid_dl.model_get_trainable_list(model_keras)

                trainable_original, layer_names = self.trainable_original, self.layer_names

                self.do_list_original = aid_dl.get_dropout(model_keras)#Get a list of dropout values of the current model

                do_list_original = self.do_list_original

            if collection==True: #if there is a collection of models:
                #Original learning rate (before expert mode is switched on!)
                self.learning_rate_original = [model_keras[1][i].optimizer.get_config()["learning_rate"] for i in range(len(model_keras[1]))]
                #Get initial trainability states of model
                trainable_layerName = [aid_dl.model_get_trainable_list(model_keras[1][i]) for i in range(len(model_keras[1]))]
                self.trainable_original = [trainable_layerName[i][0] for i in range(len(trainable_layerName))]
                self.layer_names = [trainable_layerName[i][1] for i in range(len(trainable_layerName))]
                trainable_original, layer_names = self.trainable_original, self.layer_names
                self.do_list_original = [aid_dl.get_dropout(model_keras[1][i]) for i in range(len(model_keras[1]))]#Get a list of dropout values of the current model
                do_list_original = self.do_list_original

            #TODO add expert mode ability for collection of models. Maybe define self.model_keras as a list in general. So, fitting a single model is just a special case


# =============================================================================
#             if expert_mode==True:
#                 #Apply the changes to trainable states:
#                 if train_last_layers==True:#Train only the last n layers
#                     print("Train only the last "+str(train_last_layers_n)+ " layer(s)")
#                     trainable_new = (len(trainable_original)-train_last_layers_n)*[False]+train_last_layers_n*[True]
#                     aid_dl.model_change_trainability(model_keras,trainable_new,model_metrics,out_dim,loss_expert,optimizer_settings,learning_rate_const)
#
#                 if train_dense_layers==True:#Train only dense layers
#                     print("Train only dense layers")
#                     layer_dense_ind = ["Dense" in x for x in layer_names]
#                     layer_dense_ind = np.where(np.array(layer_dense_ind)==True)[0] #at which indices are dropout layers?
#                     #create a list of trainable states
#                     trainable_new = len(trainable_original)*[False]
#                     for index in layer_dense_ind:
#                         trainable_new[index] = True
#                     aid_dl.model_change_trainability(model_keras,trainable_new,model_metrics,out_dim,loss_expert,optimizer_settings,learning_rate_const)
#
#                 if dropout_expert_on==True:
#                     #The user apparently want to change the dropout rates
#                     do_list = aid_dl.get_dropout(model_keras)#Get a list of dropout values of the current model
#                     #Compare the dropout values in the model to the dropout values requested by user
#                     if len(dropout_expert)==1:#if the user gave a float
#                         dropout_expert_list=len(do_list)*dropout_expert #convert to list
#                     elif len(dropout_expert)>1:
#                         dropout_expert_list = dropout_expert
#                         if not len(dropout_expert_list)==len(do_list):
#                             msg = QtWidgets.QMessageBox()
#                             msg.setIcon(QtWidgets.QMessageBox.Information)
#                             msg.setText("Issue with dropout: you defined "+str(len(dropout_expert_list))+" dropout rates, but model has "+str(len(do_list))+" dropout layers")
#                             msg.setWindowTitle("Issue with Expert->Dropout")
#                             msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
#                             msg.exec_()
#                             dropout_expert_list = []
#                             return
#                     else:
#                         msg = QtWidgets.QMessageBox()
#                         msg.setIcon(QtWidgets.QMessageBox.Information)
#                         msg.setText("Could not understand user input at Expert->Dropout")
#                         msg.setWindowTitle("Issue with Expert->Dropout")
#                         msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
#                         msg.exec_()
#
#                         dropout_expert_list = []
#
#                     if len(dropout_expert_list)>0 and do_list!=dropout_expert_list:#if the dropout rates of the current model is not equal to the required do_list from user...
#                         do_changed = aid_dl.change_dropout(model_keras,dropout_expert_list,model_metrics_t,nr_classes,loss_expert,optimizer_settings,learning_rate_const)
#                         if do_changed==1:
#                             text_do = "Dropout rate(s) in model was/were changed to: "+str(dropout_expert_list)
#                         else:
#                             text_do = "Dropout rate(s) in model was/were not changed"
#                     else:
#                         text_do = "Dropout rate(s) in model was/were not changed"
#                     print(text_do)
# =============================================================================


            text_updates = ""
            #Learning Rate: Compare current lr and the lr on expert tab:
            if collection == False:
                lr_current = model_keras.optimizer.get_config()["learning_rate"]
            else:
                lr_current = model_keras[1][0].optimizer.get_config()["learning_rate"]
            lr_diff = learning_rate_const-lr_current
            if  abs(lr_diff) > 1e-6: #If there is a difference, change lr accordingly
                K.set_value(model_keras.optimizer.lr, learning_rate_const)
            text_updates += "Learning rate: "+str(lr_current)+"\n"

            recompile = False
            #Compare current optimizer and the optimizer on expert tab:
            if collection==False:
                optimizer_current = aid_dl.get_optimizer_name(model_keras).lower()#get the current optimizer of the model
            else:
                optimizer_current = aid_dl.get_optimizer_name(model_keras[1][0]).lower()#get the current optimizer of the model

            if optimizer_current!=optimizer_expert.lower():#if the current model has a different optimizer
                recompile = True
            text_updates+="Optimizer: "+optimizer_expert+"\n"

            #Loss function: Compare current loss function and the loss-function on expert tab:
            if collection==False:
                if model_keras.loss!=loss_expert:
                    recompile = True
            if collection==True:
                if model_keras[1][0].loss!=loss_expert:
                    recompile = True
            text_updates += "Loss function: "+loss_expert+"\n"

            if recompile==True:
                if collection==False:
                    print("Recompiling...")
                    model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                    aid_dl.model_compile(model_keras,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
                if collection==True:
                    for m in model_keras[1]:
                        print("Recompiling...")
                        model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                        aid_dl.model_compile(m,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
            self.model_keras = model_keras #overwrite the model in self

            if collection == False:
                #Get user-specified filename for the new model
                new_modelname = str(self.lineEdit_modelname.text())
                if len(new_modelname)>0:
                    text_new_modelname = "Model will be saved as: "+new_modelname+"\n"
                else:
                    text_new_modelname = "Please specify a model path (name for the model to be fitted)\n"

            if collection == True:
                new_modelname = str(self.lineEdit_modelname.text())
                if len(new_modelname)>0:
                    new_modelname = os.path.split(new_modelname)
                    text_new_modelname = "Collection of Models will be saved into: "+new_modelname[0]+"\n"
                else:
                    text_new_modelname = "Please specify a model path (name for the model to be fitted)\n"


            #Info about normalization method
            norm = str(self.comboBox_Normalization.currentText())

            text4 = "Input image normalization method: "+norm+"\n"

            #Check if there are dropout layers:
            #do_list = aid_dl.get_dropout(model_keras)#Get a list of dropout values of the current model
            if len(do_list_original)>0:
                text4 = text4+"Found "+str(len(do_list_original)) +" dropout layers with rates: "+str(do_list_original)+"\n"
            else:
                text4 = text4+"Found no dropout layers\n"

# =============================================================================
#             if expert_mode==True:
#                 if dropout_expert_on:
#                     text4 = text4+text_do+"\n"
# =============================================================================

            text5 = "Model summary:\n"
            summary = []
            if collection==False:
                model_keras.summary(print_fn=summary.append)
                summary = "\n".join(summary)
                text = text_new_modelname+text0+text1+text2+text3+text4+text_updates+text5+summary
                self.textBrowser_Info.setText(text)

                #Save the model architecture: serialize to JSON
                model_json = model_keras.to_json()
                with open(new_modelname.split(".model")[0]+".arch", "w") as json_file:
                    json_file.write(model_json)

            elif collection==True:
                if self.groupBox_expertMode.isChecked()==True:
                    self.groupBox_expertMode.setChecked(False)
                    print("Turned off expert mode. Not implemented yet for collections of models. This does not affect user-specified metrics (precision/recall/auc)")

                self.model_keras_arch_path = [new_modelname[0]+os.sep+new_modelname[1].split(".model")[0]+"_"+model_keras[0][i]+".arch" for i in range(len(model_keras[0]))]
                for i in range(len(model_keras[1])):
                    model_keras[1][i].summary(print_fn=summary.append)

                    #Save the model architecture: serialize to JSON
                    model_json = model_keras[1][i].to_json()
                    with open(self.model_keras_arch_path[i], "w") as json_file:
                        json_file.write(model_json)

                summary = "\n".join(summary)
                text = text_new_modelname+text0+text1+text2+text3+text4+text_updates+text5+summary
                self.textBrowser_Info.setText(text)

            #Save the model to a variable on self
            self.model_keras = model_keras

            #Get the user-defined cropping size
            crop = int(self.spinBox_imagecrop.value())
            #Make the cropsize a bit larger since the images will later be rotated
            cropsize2 = np.sqrt(crop**2+crop**2)
            cropsize2 = np.ceil(cropsize2 / 2.) * 2 #round to the next even number

            #Estimate RAM needed
            nr_imgs = np.sum([np.array(list(SelectedFiles)[i]["nr_images"]) for i in range(len(list(SelectedFiles)))])
            ram_needed = np.round(nr_imgs * aid_bin.calc_ram_need(cropsize2),2)

            if duties=="initialize":#Stop here if the model just needs to be intialized (for expert mode->partial trainability)
                return

            elif duties=="initialize_train":
                #Tell the user if the data is stored and read from ram or not
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Question)
                text = "<html><head/><body><p>Should the model only be initialized,\
                or do you want to start fitting right after? For fitting, data will\
                be loaded to RAM (since Edit->Data to RAM is enabled), which will\
                require "+str(ram_needed)+"MB of RAM.</p></body></html>"
                msg.setText(text)
                msg.setWindowTitle("Initialize model or initialize and fit model?")
                msg.addButton(pg.Qt.QtGui.QPushButton('Stop after model initialization'), pg.Qt.QtGui.QMessageBox.RejectRole)
                msg.addButton(pg.Qt.QtGui.QPushButton('Start fitting'), pg.Qt.QtGui.QMessageBox.ApplyRole)
                retval = msg.exec_()

            elif duties=="initialize_lrfind":
                retval = 1

            else:
                print("Invalid duties: "+duties)
                return

            if retval==0: #yes role: Only initialize model
                print("Closing session")
                del model_keras
                sess.close()
                return

            elif retval == 1:
# =============================================================================
#                 if self.actionDataToRam.isChecked():
#                     color_mode = self.get_color_mode()
#                     zoom_factors = [selectedfile["zoom_factor"] for selectedfile in SelectedFiles]
#                     #zoom_order = [self.actionOrder0.isChecked(),self.actionOrder1.isChecked(),self.actionOrder2.isChecked(),self.actionOrder3.isChecked(),self.actionOrder4.isChecked(),self.actionOrder5.isChecked()]
#                     #zoom_order = int(np.where(np.array(zoom_order)==True)[0])
#                     zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order
#
#                     #Check if there is data already available in RAM
#                     if len(self.ram)==0:#if there is already data stored on ram
#                         print("No data on RAM. I have to load")
#                         dic = aid_img.crop_imgs_to_ram(list(SelectedFiles),cropsize2,zoom_factors=zoom_factors,zoom_order=zoom_order,color_mode=color_mode)
#                         self.ram = dic
#
#                     else:
#                         print("There is already some data on RAM")
#                         new_fileinfo = {"SelectedFiles":list(SelectedFiles),"cropsize2":cropsize2,"zoom_factors":zoom_factors,"zoom_order":zoom_order,"color_mode":color_mode}
#                         identical = aid_bin.ram_compare_data(self.ram,new_fileinfo)
#                         if not identical:
#                             #Load the data
#                             dic = aid_img.crop_imgs_to_ram(list(SelectedFiles),cropsize2,zoom_factors=zoom_factors,zoom_order=zoom_order,color_mode=color_mode)
#                             self.ram = dic
#                         if identical:
#                             msg = QtWidgets.QMessageBox()
#                             msg.setIcon(QtWidgets.QMessageBox.Question)
#                             text = "Data was loaded before! Should same data be reused? If not, click 'Reload data', e.g. if you altered the Data-table."
#                             text = "<html><head/><body><p>"+text+"</p></body></html>"
#                             msg.setText(text)
#                             msg.setWindowTitle("Found data on RAM")
#                             msg.addButton(QtGui.QPushButton('Reuse data'), QtGui.QMessageBox.YesRole)
#                             msg.addButton(QtGui.QPushButton('Reload data'), QtGui.QMessageBox.NoRole)
#                             retval = msg.exec_()
#
#                             if retval==0:
#                                 print("Re-use data")
#                                 #Re-use same data
#                             elif retval==1:
#                                 print("Re-load data")
#                                 dic = aid_img.crop_imgs_to_ram(list(SelectedFiles),cropsize2,zoom_factors=zoom_factors,zoom_order=zoom_order,color_mode=color_mode)
#                                 self.ram = dic
#
# =============================================================================
                #Finally, activate the 'Fit model' button again
                #self.pushButton_FitModel.setEnabled(True)
                if duties=="initialize_train":
                    self.action_fit_model()
                if duties=="initialize_lrfind":
                    self.action_lr_finder()

            del model_keras

    def popup_normalization(self):
            self.w = MyPopup()
            self.gridLayout_w = QtWidgets.QGridLayout(self.w)
            self.gridLayout_w.setObjectName(_fromUtf8("gridLayout"))
            self.verticalLayout_w = QtWidgets.QVBoxLayout()
            self.verticalLayout_w.setObjectName(_fromUtf8("verticalLayout"))
            self.label_w = QtWidgets.QLabel(self.w)
            self.label_w.setAlignment(QtCore.Qt.AlignCenter)
            self.label_w.setObjectName(_fromUtf8("label_w"))
            self.verticalLayout_w.addWidget(self.label_w)
            self.horizontalLayout_2_w = QtWidgets.QHBoxLayout()
            self.horizontalLayout_2_w.setObjectName(_fromUtf8("horizontalLayout_2"))
            self.pushButton_w = QtWidgets.QPushButton(self.w)
            self.pushButton_w.setObjectName(_fromUtf8("pushButton"))
            self.horizontalLayout_2_w.addWidget(self.pushButton_w)
            self.horizontalLayout_w = QtWidgets.QHBoxLayout()
            self.horizontalLayout_w.setObjectName(_fromUtf8("horizontalLayout"))
            self.label_2_w = QtWidgets.QLabel(self.w)
            self.label_2_w.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.label_2_w.setObjectName(_fromUtf8("label_2_w"))
            self.horizontalLayout_w.addWidget(self.label_2_w)
            self.comboBox_w = QtWidgets.QComboBox(self.w)
            self.comboBox_w.setObjectName(_fromUtf8("comboBox"))
            self.comboBox_w.addItems(["Select"]+self.norm_methods)
            self.comboBox_w.setMinimumSize(200,22)
            self.comboBox_w.setMaximumSize(200, 22)
            width=self.comboBox_w.fontMetrics().boundingRect(max(self.norm_methods, key=len)).width()
            self.comboBox_w.view().setFixedWidth(width+10)
            self.comboBox_w.currentIndexChanged.connect(self.get_norm_from_manualselection)
            self.horizontalLayout_w.addWidget(self.comboBox_w)
            self.horizontalLayout_2_w.addLayout(self.horizontalLayout_w)
            self.verticalLayout_w.addLayout(self.horizontalLayout_2_w)
            self.gridLayout_w.addLayout(self.verticalLayout_w, 0, 0, 1, 1)

            self.w.setWindowTitle("Select normalization method")
            self.label_w.setText("You are about to continue training a pretrained model\n"
    "Please select the meta file of that model to load the normalization method\n"
    "or choose the normalization method manually")
            self.pushButton_w.setText("Load meta file")
            self.label_2_w.setText("Manual \n"
    "selection")

            #one button that allows to load a meta file containing the norm-method
            self.pushButton_w.clicked.connect(self.get_norm_from_modelparafile)
            self.w.show()


    def action_fit_model(self):
        #Take the initialized model
        #Unfortunately, in TensorFlow it is not possile to pass a model from
        #one thread to another. Therefore I have to load and save the models each time :(
        model_keras = self.model_keras
        if type(model_keras)==tuple:
            collection=True
        else:
            collection=False

        #Check if there was a model initialized:
        new_modelname = str(self.lineEdit_modelname.text())

        if len(new_modelname)==0:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Please define a path/filename for the model to be fitted!")
           msg.setWindowTitle("Model path/ filename missing!")
           msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
           msg.exec_()
           return

        if model_keras==None:#in case the model got deleted in another task
            self.action_initialize_model(duties="initialize_train")

            print("Had to re-run action_initialize_model!")
            model_keras = self.model_keras
            self.model_keras = None#delete this copy

            if model_keras==None:
#                msg = QtWidgets.QMessageBox()
#                msg.setIcon(QtWidgets.QMessageBox.Information)
#                msg.setText("Model could not be initialized")
#                msg.setWindowTitle("Error")
#                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
#                msg.exec_()
                return
            if not model_keras==None:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText("Model is now initialized for you, Please check Model summary window below if everything is correct and then press Fit again!")
                msg.setWindowTitle("No initilized model found!")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
                return

        #There should be at least two outputs (index 0 and 1)
        if collection==False:
            #model_config = model_keras.aid_dl.get_config()#["layers"]
            nr_classes = int(model_keras.output.shape.dims[1])

        if collection==True:
            #model_config = model_keras[1][0].aid_dl.get_config()#["layers"]
            nr_classes = int(model_keras[1][0].output.shape.dims[1])

        if nr_classes<2:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Please define at least two classes")
           msg.setWindowTitle("Not enough classes")
           msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
           msg.exec_()
           return

        if collection==False:
            #define a variable on self which allows the fit_model_worker to load this model and fit
            #(sorry, this is necessary since TensorFlow does not support passing models between threads)
            self.model_keras_path = new_modelname.split(".model")[0]+"_0.model"
            #save a first version of the .model
            model_keras.save(self.model_keras_path,save_format='h5')
            #Delete the variable to save RAM
            model_keras = None #Since this uses TensorFlow, I have to reload the model action_fit_model_worker anyway

        if collection==True:
            #define a variable on self which allows the fit_model_worker to load this model and fit
            #(sorry, this is necessary since TensorFlow does not support passing models between threads)
            self.model_keras_path = [new_modelname.split(".model")[0]+"_"+model_keras[0][i]+".model" for i in range(len(model_keras[0]))]
            for i in range(len(self.model_keras_path)):
                #save a first version of the .model
                model_keras[1][i].save(self.model_keras_path[i])

            #Delete the variable to save RAM
            model_keras = None #Since this uses TensorFlow, I have to reload the model action_fit_model_worker anyway
        #Check that Data is on RAM
        DATA_len = len(self.ram) #this returns the len of a dictionary. The dictionary is supposed to contain the training/validation data; otherwise the data is read from .rtdc data directly (SLOW unless you have ultra-good SSD)

        def popup_data_to_ram(button):
            yes_or_no = button.text()
            if yes_or_no == "&Yes":
                print("Moving data to ram")
                self.actionDataToRamNow_function()
            elif yes_or_no == "&No":
                pass

        if DATA_len==0:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Would you like transfer the Data to RAM now?\n(Currently the data is not in RAM and would be read from .rtdc, which slows down fitting dramatically unless you have a super-fast SSD.)")
           msg.setWindowTitle("Data to RAM now?")
           msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
           msg.buttonClicked.connect(popup_data_to_ram)
           msg.exec_()

        ###################Popup Window####################################
        self.fittingpopups.append(MyPopup())
        ui = Fitting_Ui()
        ui.setupUi(self.fittingpopups[-1]) #append the ui to the last element on the list
        self.fittingpopups_ui.append(ui)
        # Increase the popupcounter by one; this will help to coordinate the data flow between main ui and popup
        self.popupcounter += 1
        listindex=self.popupcounter-1

        ##############################Define functions#########################
        self.fittingpopups_ui[listindex].pushButton_UpdatePlot_pop.clicked.connect(lambda: self.update_historyplot_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_Stop_pop.clicked.connect(lambda: self.stop_fitting_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_Pause_pop.clicked.connect(lambda: self.pause_fitting_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_saveTextWindow_pop.clicked.connect(lambda: self.saveTextWindow_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_clearTextWindow_pop.clicked.connect(lambda: self.clearTextWindow_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_showModelSumm_pop.clicked.connect(lambda: self.showModelSumm_pop(listindex))
        self.fittingpopups_ui[listindex].pushButton_saveModelSumm_pop.clicked.connect(lambda: self.saveModelSumm_pop(listindex))
        #Expert mode functions
        #self.fittingpopups_ui[listindex].checkBox_pTr_pop.toggled.connect(lambda on_or_off: self.partialtrainability_activated_pop(on_or_off,listindex))

        self.fittingpopups_ui[listindex].Form.setWindowTitle(os.path.split(new_modelname)[1])
        self.fittingpopups_ui[listindex].progressBar_Fitting_pop.setValue(0) #set the progress bar to zero
        self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.doubleClicked.connect(lambda item: self.tableWidget_HistoryInfo_pop_dclick(item,listindex))



        worker = Worker(self.action_fit_model_worker)
        #Get a signal from the worker to update the progressbar
        worker.signals.progress.connect(self.fittingpopups_ui[listindex].progressBar_Fitting_pop.setValue)

        #Define a func which prints information during fitting to textbrowser
        #And furthermore provide option to do real-time plotting
        def real_time_info(dic):
            self.fittingpopups_ui[listindex].Histories.append(dic) #append to a list. Will be used for plotting in the "Update plot" function
            OtherMetrics_keys = self.fittingpopups_ui[listindex].RealTime_OtherMetrics.keys()

            #Append to lists for real-time plotting
            self.fittingpopups_ui[listindex].RealTime_Acc.append(dic["accuracy"][0])
            self.fittingpopups_ui[listindex].RealTime_ValAcc.append(dic["val_accuracy"][0])
            self.fittingpopups_ui[listindex].RealTime_Loss.append(dic["loss"][0])
            self.fittingpopups_ui[listindex].RealTime_ValLoss.append(dic["val_loss"][0])


            keys = list(dic.keys())
            #sort keys alphabetically
            keys_ = [l.lower() for l in keys]
            ind_sort = np.argsort(keys_)
            keys = list(np.array(keys)[ind_sort])
            #First keys should always be acc,loss,val_acc,val_loss -in this order
            keys_first = ["accuracy","loss","val_accuracy","val_loss"]
            for i in range(len(keys_first)):
                if keys_first[i] in keys:
                    ind = np.where(np.array(keys)==keys_first[i])[0][0]
                    if ind!=i:
                        del keys[ind]
                        keys.insert(i,keys_first[i])

            for key in keys:
                if "precision" in key or "auc" in key or "recall" in key or "LearningRate" in key:
                    if not key in OtherMetrics_keys: #if this key is missing in self.fittingpopups_ui[listindex].RealTime_OtherMetrics attach it!
                        self.fittingpopups_ui[listindex].RealTime_OtherMetrics[key] = []
                    self.fittingpopups_ui[listindex].RealTime_OtherMetrics[key].append(dic[key])
            dic_text = [(f"{item} {np.round(amount[0],4)}") for item, amount in dic.items()]
            text = "Epoch "+str(self.fittingpopups_ui[listindex].epoch_counter)+"\n"+" ".join(dic_text)
            self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)
            self.fittingpopups_ui[listindex].epoch_counter+=1
            if self.fittingpopups_ui[listindex].epoch_counter==1:

                #for each key, put a checkbox on the tableWidget_HistoryInfo_pop
                rowPosition = self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.rowCount()
                self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.insertRow(rowPosition)
                self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.setColumnCount(len(keys))

                for columnPosition in range(len(keys)):#(2,4):
                    key = keys[columnPosition]
                    #for each item, also create 2 checkboxes (train/valid)
                    item = QtWidgets.QTableWidgetItem(str(key))#("item {0} {1}".format(rowNumber, columnNumber))
                    item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))# set text color: default black
                    item.setBackground(QtGui.QColor(self.colorsQt[columnPosition]))
                    item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
                    item.setCheckState(QtCore.Qt.Unchecked)
                    self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.setItem(rowPosition, columnPosition, item)

            self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.resizeColumnsToContents()
            self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.resizeRowsToContents()


            ########################Real-time plotting#########################
            if self.fittingpopups_ui[listindex].checkBox_realTimePlotting_pop.isChecked():
                #get the range for the real time fitting
                if hasattr(self.fittingpopups_ui[listindex], 'historyscatters'):#if update plot was hit before
                    x = range(len(self.fittingpopups_ui[listindex].Histories))
                    realTimeEpochs = self.fittingpopups_ui[listindex].spinBox_realTimeEpochs.value()
                    if len(x)>realTimeEpochs:
                        x = x[-realTimeEpochs:]
                    #is any metric checked on the table?
                    colcount = int(self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.columnCount())
                    #Collect items that are checked
                    selected_items,Colors = [],[]
                    for colposition in range(colcount):
                        #is it checked?
                        cb = self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.item(0, colposition)
                        if not cb==None:
                            if cb.checkState() == QtCore.Qt.Checked:
                                selected_items.append(str(cb.text()))
                                Colors.append(cb.background())

                    for i in range(len(self.fittingpopups_ui[listindex].historyscatters)): #iterate over all available plots
                        key = list(self.fittingpopups_ui[listindex].historyscatters.keys())[i]
                        if key in selected_items:
                            if key=="accuracy":
                                y = np.array(self.fittingpopups_ui[listindex].RealTime_Acc).astype(float)
                            elif key=="val_accuracy":
                                y = np.array(self.fittingpopups_ui[listindex].RealTime_ValAcc).astype(float)
                            elif key=="loss":
                                y = np.array(self.fittingpopups_ui[listindex].RealTime_Loss).astype(float)
                            elif key=="val_loss":
                                y = np.array(self.fittingpopups_ui[listindex].RealTime_ValLoss).astype(float)
                            elif "precision" in key or "auc" in key or "recall" in key or "LearningRate" in key:
                               y = np.array(self.fittingpopups_ui[listindex].RealTime_OtherMetrics[key]).astype(float).reshape(-1,)
                            else:
                                return
                            #Only show the last 250 epochs
                            if y.shape[0]>realTimeEpochs:
                                y = y[-realTimeEpochs:]
                            if y.shape[0]==len(x):
                                self.fittingpopups_ui[listindex].historyscatters[key].setData(x, y)#,pen=None,symbol='o',symbolPen=None,symbolBrush=brush,clear=False)
                            else:
                                print("x and y are not the same size! Omitted plotting. I will try again to plot after the next epoch.")

                        pg.QtGui.QApplication.processEvents()

        self.fittingpopups_ui[listindex].epoch_counter = 0
        #self.fittingpopups_ui[listindex].backup = [] #backup of the meta information -> in case the original folder is not accessible anymore
        worker.signals.history.connect(real_time_info)

        #Finally start the worker!
        self.threadpool.start(worker)
        self.fittingpopups[listindex].show()



    def action_lr_finder(self):
        #lr_find
        model_keras = self.model_keras
        if type(model_keras)==tuple:
            collection=True
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("LR screening is not supported for Collections of models. Please select single model")
            msg.setWindowTitle("LR screening not supported for Collections!")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return
        else:
            collection=False

        #Check if there was a model initialized:
        new_modelname = str(self.lineEdit_modelname.text())

        if len(new_modelname)==0:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Please define a path/filename for the model to be fitted!")
           msg.setWindowTitle("Model path/ filename missing!")
           msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
           msg.exec_()
           return

        if model_keras==None:#in case the model got deleted in another task
            self.action_initialize_model(duties="initialize_train")
            print("Had to re-run action_initialize_model!")
            model_keras = self.model_keras
            self.model_keras = None#delete this copy

            if model_keras==None:
                return
            if not model_keras==None:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText("Model is now initialized for you, Please check Model summary window below if everything is correct and then press Fit again!")
                msg.setWindowTitle("No initilized model found!")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
                return

        nr_classes = int(model_keras.output.shape.dims[1])

        if nr_classes<2:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Please define at least two classes")
           msg.setWindowTitle("Not enough classes")
           msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
           msg.exec_()
           return

        #define a variable on self which allows the fit_model_worker to load this model and fit
        #(sorry, this is necessary since TensorFlow does not support passing models between threads)
        self.model_keras_path = new_modelname.split(".model")[0]+"_0.model"
        #save a first version of the .model
        model_keras.save(self.model_keras_path,save_format='h5')
        #Delete the variable to save RAM
        model_keras = None #Since this uses TensorFlow, I have to reload the model action_fit_model_worker anyway

        #Check that Data is on RAM
        DATA_len = len(self.ram) #this returns the len of a dictionary. The dictionary is supposed to contain the training/validation data; otherwise the data is read from .rtdc data directly (SLOW unless you have ultra-good SSD)

        def popup_data_to_ram(button):
            yes_or_no = button.text()
            if yes_or_no == "&Yes":
                print("Moving data to ram")
                self.actionDataToRamNow_function()
            elif yes_or_no == "&No":
                pass

        if DATA_len==0:
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)
           msg.setText("Would you like transfer the Data to RAM now?\n(Currently the data is not in RAM and would be read from .rtdc, which slows down fitting dramatically unless you have a super-fast SSD.)")
           msg.setWindowTitle("Data to RAM now?")
           msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
           msg.buttonClicked.connect(popup_data_to_ram)
           msg.exec_()

        worker = Worker(self.action_lr_finder_worker)
        #Get a signal from the worker to update the progressbar
        worker.signals.progress.connect(print)
        worker.signals.history.connect(print)
        #Finally start the worker!
        self.threadpool.start(worker)


    def action_fit_model_worker(self,progress_callback,history_callback):
        if self.radioButton_cpu.isChecked():
            gpu_used = False
            deviceSelected = str(self.comboBox_cpu.currentText())
        elif self.radioButton_gpu.isChecked():
            gpu_used = True
            deviceSelected = str(self.comboBox_gpu.currentText())
        gpu_memory = float(self.doubleSpinBox_memory.value())

        #Create config (define which device to use)
        config_gpu = aid_dl.get_config(cpu_nr,gpu_nr,deviceSelected,gpu_memory)

        with tf.compat.v1.Session(graph = tf.Graph(), config=config_gpu) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            #get an index of the fitting popup
            listindex = self.popupcounter-1
            #Get user-specified filename for the new model
            new_modelname = str(self.lineEdit_modelname.text())
            model_keras_path = self.model_keras_path

            if type(model_keras_path)==list:
                collection = True
                #Take the initialized models
                model_keras_path = self.model_keras_path
                model_keras = [load_model(model_keras_path[i],custom_objects=aid_dl.get_custom_metrics()) for i in range(len(model_keras_path)) ]
                model_architecture_names = self.model_keras[0]
                print(model_architecture_names)
                #self.model_keras = None

            else:
                collection = False

# =============================================================================
#                 if deviceSelected=="Multi-GPU" and cpu_weight_merge==True:
#                     strategy = tf.distribute.MirroredStrategy()
#                     with strategy.scope():
#                         model_keras = load_model(model_keras_path,custom_objects=aid_dl.get_custom_metrics())
#                 else:
# =============================================================================
            model_keras = load_model(model_keras_path,custom_objects=aid_dl.get_custom_metrics())

            #Initialize a variable for the parallel model
            model_keras_p = None


            ##############Main function after hitting FIT MODEL####################
            if self.radioButton_LoadRestartModel.isChecked():
                load_modelname = str(self.lineEdit_LoadModelPath.text())
            elif self.radioButton_LoadContinueModel.isChecked():
                load_modelname = str(self.lineEdit_LoadModelPath.text())
            elif self.radioButton_NewModel.isChecked():
                load_modelname = "" #No model is loaded

            if collection==False:
                #model_config = model_keras.aid_dl.get_config()#["layers"]
                nr_classes = int(model_keras.output.shape.dims[1])
            if collection==True:
                #model_config = model_keras.aid_dl.get_config()#["layers"]
                nr_classes = int(model_keras[0].output.shape.dims[1])

            #Metrics to be displayed during fitting (real-time)
            model_metrics = self.get_metrics()
            model_metrics_t = aid_dl.get_metrics_tensors(model_metrics,nr_classes)

            #Compile model
            if collection==False and deviceSelected=="Single-GPU":
                model_keras.compile(loss='categorical_crossentropy',optimizer='adam',metrics=aid_dl.get_metrics_tensors(model_metrics,nr_classes))#dont specify loss and optimizer yet...expert stuff will follow and model will be recompiled
            elif collection==False and deviceSelected=="Multi-GPU":
                model_keras_p.compile(loss='categorical_crossentropy',optimizer='adam',metrics=aid_dl.get_metrics_tensors(model_metrics,nr_classes))#dont specify loss and optimizer yet...expert stuff will follow and model will be recompiled
            elif collection==True and deviceSelected=="Single-GPU":
                for m in model_keras:
                    model_metrics_ = self.get_metrics()
                    m.compile(loss='categorical_crossentropy',optimizer='adam',metrics=aid_dl.get_metrics_tensors(model_metrics_,nr_classes))#dont specify loss and optimizer yet...expert stuff will follow and model will be recompiled
            elif collection==True and deviceSelected=="Multi-GPU":
                print("Collection & Multi-GPU is not supported yet")
                return

            #Original learning rate:
            #learning_rate_original = self.learning_rate_original#K.eval(model_keras.optimizer.lr)
            #Original trainable states of layers with parameters
            trainable_original, layer_names = self.trainable_original, self.layer_names
            do_list_original = self.do_list_original

            #Collect all information about the fitting routine that was user
            #defined
# =============================================================================
#             if self.actionVerbose.isChecked()==True:
#                 verbose = 1
#             else:
# =============================================================================
            verbose = 0

            new_model = self.radioButton_NewModel.isChecked()
            chosen_model = str(self.comboBox_ModelSelection.currentText())

            crop = int(self.spinBox_imagecrop.value())
            color_mode = str(self.comboBox_GrayOrRGB.currentText())

            loadrestart_model = self.radioButton_LoadRestartModel.isChecked()
            loadcontinue_model = self.radioButton_LoadContinueModel.isChecked()

            norm = str(self.comboBox_Normalization.currentText())

            nr_epochs = int(self.spinBox_NrEpochs.value())
            keras_refresh_nr_epochs = int(self.spinBox_RefreshAfterEpochs.value())
            h_flip = bool(self.checkBox_HorizFlip.isChecked())
            v_flip = bool(self.checkBox_VertFlip.isChecked())
            rotation = float(self.lineEdit_Rotation.text())

            width_shift = float(self.lineEdit_widthShift.text())
            height_shift = float(self.lineEdit_heightShift.text())
            zoom = float(self.lineEdit_zoomRange.text())
            shear = float(self.lineEdit_shearRange.text())

            brightness_refresh_nr_epochs = int(self.spinBox_RefreshAfterEpochs.value())
            brightness_add_lower = float(self.spinBox_PlusLower.value())
            brightness_add_upper = float(self.spinBox_PlusUpper.value())
            brightness_mult_lower = float(self.doubleSpinBox_MultLower.value())
            brightness_mult_upper = float(self.doubleSpinBox_MultUpper.value())
            gaussnoise_mean = float(self.doubleSpinBox_GaussianNoiseMean.value())
            gaussnoise_scale = float(self.doubleSpinBox_GaussianNoiseScale.value())

            contrast_on = bool(self.checkBox_contrast.isChecked())
            contrast_lower = float(self.doubleSpinBox_contrastLower.value())
            contrast_higher = float(self.doubleSpinBox_contrastHigher.value())
            saturation_on = bool(self.checkBox_saturation.isChecked())
            saturation_lower = float(self.doubleSpinBox_saturationLower.value())
            saturation_higher = float(self.doubleSpinBox_saturationHigher.value())
            hue_on = bool(self.checkBox_hue.isChecked())
            hue_delta = float(self.doubleSpinBox_hueDelta.value())

            avgBlur_on = bool(self.checkBox_avgBlur.isChecked())
            avgBlur_min = int(self.spinBox_avgBlurMin.value())
            avgBlur_max = int(self.spinBox_avgBlurMax.value())
            gaussBlur_on = bool(self.checkBox_gaussBlur.isChecked())
            gaussBlur_min = int(self.spinBox_gaussBlurMin.value())
            gaussBlur_max = int(self.spinBox_gaussBlurMax.value())
            motionBlur_on = bool(self.checkBox_motionBlur.isChecked())
            motionBlur_kernel = str(self.lineEdit_motionBlurKernel.text())
            motionBlur_angle = str(self.lineEdit_motionBlurAngle.text())
            motionBlur_kernel = tuple(ast.literal_eval(motionBlur_kernel)) #translate string in the lineEdits to a tuple
            motionBlur_angle = tuple(ast.literal_eval(motionBlur_angle)) #translate string in the lineEdits to a tuple

            expert_mode = False

            batchSize_expert = 32
            epochs_expert = 1
            learning_rate_expert_on = False
            learning_rate_const_on = False
            learning_rate_const = 0.001
            learning_rate_cycLR_on = False


            cycLrMin = []
            cycLrMax = []
            cycLrMethod = "triangular"
            #clr_settings = self.fittingpopups_ui[listindex].clr_settings.copy()
            cycLrGamma = self.clr_settings["gamma"]
            SelectedFiles = self.items_clicked()#to compute cycLrStepSize, the number of training images is needed
            cycLrStepSize = aid_dl.get_cyclStepSize(SelectedFiles,self.clr_settings["step_size"],batchSize_expert)
            #put clr_settings onto fittingpopup,
            self.fittingpopups_ui[listindex].clr_settings = self.clr_settings.copy()#assign a copy. Otherwise values in both dicts are changed when manipulating one dict
            #put optimizer_settings onto fittingpopup,
            self.fittingpopups_ui[listindex].optimizer_settings = self.optimizer_settings.copy()#assign a copy. Otherwise values in both dicts are changed when manipulating one dict


            learning_rate_expo_on = False
            expDecInitLr = 0.001
            expDecSteps = 0
            expDecRate = 0.96

            loss_expert_on = False
            loss_expert = "categorical_crossentropy"
            optimizer_expert_on = False
            optimizer_expert = "Adam"
            optimizer_settings = self.fittingpopups_ui[listindex].optimizer_settings.copy()#make a copy to make sure that changes in the UI are not immediately used

            paddingMode = str(self.comboBox_paddingMode.currentText())#.lower()

            train_last_layers = False
            train_last_layers_n = 0
            train_dense_layers = False
            dropout_expert_on = False


            try:
                dropout_expert = str(self.lineEdit_dropout.text()) #due to the validator, there are no squ.brackets
                dropout_expert = "["+dropout_expert+"]"
                dropout_expert = ast.literal_eval(dropout_expert)
            except:
                dropout_expert = []
            lossW_expert_on = False
            lossW_expert = ""


            #To get the class weights (loss), the SelectedFiles are required
            #SelectedFiles = self.items_clicked()
            #Check if xtra_data should be used for training
            xtra_in = [s["xtra_in"] for s in SelectedFiles]
            if len(set(xtra_in))==1:
                xtra_in = list(set(xtra_in))[0]
            elif len(set(xtra_in))>1:# False and True is present. Not supported
                print("Xtra data is used only for some files. Xtra data needs to be used either by all or by none!")
                return

            self.fittingpopups_ui[listindex].SelectedFiles = SelectedFiles #save to self. to make it accessible for popup showing loss weights
            #Get the class weights. This function runs now the first time in the fitting routine.
            #It is possible that the user chose Custom weights and then changed the classes. Hence first check if
            #there is a weight for each class available.
            class_weight = self.get_class_weight(self.fittingpopups_ui[listindex].SelectedFiles,lossW_expert,custom_check_classes=True)
            if type(class_weight)==list:
                #There has been a mismatch between the classes described in class_weight and the classes available in SelectedFiles!
                lossW_expert = class_weight[0] #overwrite
                class_weight = class_weight[1]
                print("class_weight:" +str(class_weight))
                print("There has been a mismatch between the classes described in \
                      Loss weights and the classes available in the selected files! \
                      Hence, the Loss weights are set to Balanced")

            #Get callback for the learning rate scheduling
            callback_lr = aid_dl.get_lr_callback(learning_rate_const_on,learning_rate_const,
                                               learning_rate_cycLR_on,cycLrMin,cycLrMax,
                                               cycLrMethod,cycLrStepSize,
                                               learning_rate_expo_on,
                                               expDecInitLr,expDecSteps,expDecRate,cycLrGamma)
            #save a dictionary with initial values
            lr_dict_original = aid_dl.get_lr_dict(learning_rate_const_on,learning_rate_const,
                                               learning_rate_cycLR_on,cycLrMin,cycLrMax,
                                               cycLrMethod,cycLrStepSize,
                                               learning_rate_expo_on,
                                               expDecInitLr,expDecSteps,expDecRate,cycLrGamma)

            if collection==False:
                #Create an excel file
                writer = pd.ExcelWriter(new_modelname.split(".model")[0]+'_meta.xlsx', engine='openpyxl')
                self.fittingpopups_ui[listindex].writer = writer
                #Used files go to a separate sheet on the MetaFile.xlsx
                SelectedFiles_df = pd.DataFrame(SelectedFiles)
                pd.DataFrame().to_excel(writer,sheet_name='UsedData') #initialize empty Sheet
                SelectedFiles_df.to_excel(writer,sheet_name='UsedData')
                DataOverview_df = self.get_dataOverview()
                DataOverview_df.to_excel(writer,sheet_name='DataOverview') #write data overview to separate sheet
                pd.DataFrame().to_excel(writer,sheet_name='Parameters') #initialize empty Sheet
                pd.DataFrame().to_excel(writer,sheet_name='History') #initialize empty Sheet


            elif collection==True:
                SelectedFiles_df = pd.DataFrame(SelectedFiles)

                Writers = []
                #Create excel files
                for i in range(len(model_keras_path)):
                    writer = pd.ExcelWriter(model_keras_path[i].split(".model")[0]+'_meta.xlsx', engine='openpyxl')
                    Writers.append(writer)
                for writer in Writers:
                    #Used files go to a separate sheet on the MetaFile.xlsx
                    pd.DataFrame().to_excel(writer,sheet_name='UsedData') #initialize empty Sheet
                    SelectedFiles_df.to_excel(writer,sheet_name='UsedData')
                    DataOverview_df = self.get_dataOverview()
                    DataOverview_df.to_excel(writer,sheet_name='DataOverview') #write data overview to separate sheet
                    pd.DataFrame().to_excel(writer,sheet_name='Parameters') #initialize empty Sheet
                    pd.DataFrame().to_excel(writer,sheet_name='History') #initialize empty Sheet

            ###############################Expert Mode values##################
            text_updates = ""
            #Compare current lr and the lr on expert tab:
            if collection == False:
                lr_current = model_keras.optimizer.get_config()["learning_rate"]
            else:
                lr_current = model_keras[0].optimizer.get_config()["learning_rate"]

            lr_diff = learning_rate_const-lr_current
            if  abs(lr_diff) > 1e-6:
                if collection == False:
                    K.set_value(model_keras.optimizer.lr, learning_rate_const)
                if collection == True:
                    for m in model_keras:
                        K.set_value(m.optimizer.lr, learning_rate_const)
                text_updates +=  "Changed the learning rate to "+ str(learning_rate_const)+"\n"

            #Check if model has to be compiled again
            recompile = False #by default, dont recompile (happens for "Load and continue" training a model)
            if new_model==True:
                recompile = True
            #Compare current optimizer and the optimizer on expert tab:
            if collection==False:
                optimizer_current = aid_dl.get_optimizer_name(model_keras).lower()#get the current optimizer of the model
            if collection==True:
                optimizer_current = aid_dl.get_optimizer_name(model_keras[0]).lower()#get the current optimizer of the model
            if optimizer_current!=optimizer_expert.lower():#if the current model has a different optimizer
                recompile = True
                text_updates+="Changed the optimizer to "+optimizer_expert+"\n"

            #Compare current loss function and the loss-function on expert tab:
            if collection==False:
                if model_keras.loss!=loss_expert:
                    recompile = True
                    text_updates+="Changed the loss function to "+loss_expert+"\n"
            if collection==True:
                if model_keras[0].loss!=loss_expert:
                    recompile = True
                    text_updates+="Changed the loss function to "+loss_expert+"\n"

            if recompile==True:
                print("Recompiling...")
                if collection==False:
                    aid_dl.model_compile(model_keras,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
                if collection==True:
                    for m in model_keras:
                        model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                        aid_dl.model_compile(m, loss_expert, optimizer_settings, learning_rate_const,model_metrics_t, nr_classes)
                if model_keras_p!=None:#if this is NOT None, there exists a parallel model, which also needs to be re-compiled
                    model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                    aid_dl.model_compile(model_keras_p,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
                    print("Recompiled parallel model to adjust learning rate, loss, optimizer")

            self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text_updates)

            #self.model_keras = model_keras #overwrite the model on self

            ######################Load the Training Data################################
            ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
            ind = np.where(np.array(ind)==True)[0]
            SelectedFiles_train = np.array(SelectedFiles)[ind]
            SelectedFiles_train = list(SelectedFiles_train)
            indices_train = [selectedfile["class"] for selectedfile in SelectedFiles_train]
            nr_events_epoch_train = [selectedfile["nr_events_epoch"] for selectedfile in SelectedFiles_train]
            rtdc_path_train = [selectedfile["rtdc_path"] for selectedfile in SelectedFiles_train]
            zoom_factors_train = [selectedfile["zoom_factor"] for selectedfile in SelectedFiles_train]
            zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order

            shuffle_train = [selectedfile["shuffle"] for selectedfile in SelectedFiles_train]
            xtra_in = {selectedfile["xtra_in"] for selectedfile in SelectedFiles_train}
            if len(xtra_in)>1:# False and True is present. Not supported
                print("Xtra data is used only for some files. Xtra data needs to be used either by all or by none!")
                return
            xtra_in = list(xtra_in)[0]#this is either True or False

            #read self.ram to new variable ; next clear ram. This is required for multitasking (training multiple models with maybe different data)
            DATA = self.ram
            if verbose==1:
                print("Length of DATA (in RAM) = "+str(len(DATA)))

# =============================================================================
#             #clear the ram again if desired
#             if not self.actionKeep_Data_in_RAM.isChecked():
#                 self.ram = dict()
#                 print("Removed data from self.ram. For further training sessions, data has to be reloaded.")
# =============================================================================
            self.ram = dict()
            print("Removed data from self.ram. For further training sessions, data has to be reloaded.")



            #If the scaling method is "divide by mean and std of the whole training set":
            if norm == "StdScaling using mean and std of all training data":
                mean_trainingdata,std_trainingdata = [],[]
                for i in range(len(SelectedFiles_train)):
                    #if Data_to_RAM was not enabled:
                    #if not self.actionDataToRam.isChecked():
                    if len(DATA)==0: #Here, the entire training set needs to be used! Not only random images!
                        #Replace=true: means individual cells could occur several times
                        gen_train = aid_img.gen_crop_img(crop,rtdc_path_train[i],random_images=False,zoom_factor=zoom_factors_train[i],zoom_order=zoom_order,color_mode=self.get_color_mode(),padding_mode=paddingMode)
                    else:
                        gen_train = aid_img.gen_crop_img_ram(DATA,rtdc_path_train[i],random_images=False) #Replace true means that individual cells could occur several times
# =============================================================================
#                         if self.actionVerbose.isChecked():
#                             print("Loaded data from RAM")
# =============================================================================

                    images = next(gen_train)[0]
                    mean_trainingdata.append(np.mean(images))
                    std_trainingdata.append(np.std(images))
                mean_trainingdata = np.mean(np.array(mean_trainingdata))
                std_trainingdata = np.mean(np.array(std_trainingdata))

                if np.allclose(std_trainingdata,0):
                    std_trainingdata = 0.0001

                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    text = "<html><head/><body><p>The standard deviation of your training data is zero! This would lead to division by zero. To avoid this, I will divide by 0.0001 instead.</p></body></html>"
                    msg.setText(text)
                    msg.setWindowTitle("Std. is zero")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    msg.exec_()

            Para_dict = pd.DataFrame()
            def update_para_dict():
                #Document changes in the meta-file
                Para_dict["AIDeveloper_Version"]=VERSION,
                Para_dict["model_zoo_version"]=model_zoo_version,
                try:
                    Para_dict["OS"]=platform.platform(),
                    Para_dict["CPU"]=platform.processor(),
                except:
                    Para_dict["OS"]="Unknown",
                    Para_dict["CPU"]="Unknown",

                Para_dict["Modelname"]=new_modelname,
                Para_dict["Chosen Model"]=chosen_model,
                Para_dict["new_model"]=new_model,
                Para_dict["loadrestart_model"]=loadrestart_model,
                Para_dict["loadcontinue_model"]=loadcontinue_model,
                Para_dict["Continued_Fitting_From"]=load_modelname,
                Para_dict["Input image size"]=crop,
                Para_dict["Color Mode"]=color_mode,
                Para_dict["Zoom order"]=zoom_order,
                Para_dict["Device"]=deviceSelected,
                Para_dict["gpu_used"]=gpu_used,
                Para_dict["gpu_memory"]=gpu_memory,
                Para_dict["Output Nr. classes"]=nr_classes,
                Para_dict["Normalization"]=norm,
                Para_dict["Nr. epochs"]=nr_epochs,
                Para_dict["Keras refresh after nr. epochs"]=keras_refresh_nr_epochs,
                Para_dict["Horz. flip"]=h_flip,
                Para_dict["Vert. flip"]=v_flip,
                Para_dict["rotation"]=rotation,
                Para_dict["width_shift"]=width_shift,
                Para_dict["height_shift"]=height_shift,
                Para_dict["zoom"]=zoom,
                Para_dict["shear"]=shear,
                Para_dict["Brightness refresh after nr. epochs"]=brightness_refresh_nr_epochs,
                Para_dict["Brightness add. lower"]=brightness_add_lower,
                Para_dict["Brightness add. upper"]=brightness_add_upper,
                Para_dict["Brightness mult. lower"]=brightness_mult_lower,
                Para_dict["Brightness mult. upper"]=brightness_mult_upper,
                Para_dict["Gaussnoise Mean"]=gaussnoise_mean,
                Para_dict["Gaussnoise Scale"]=gaussnoise_scale,

                Para_dict["Contrast on"]=contrast_on,
                Para_dict["Contrast Lower"]=contrast_lower,
                Para_dict["Contrast Higher"]=contrast_higher,
                Para_dict["Saturation on"]=saturation_on,
                Para_dict["Saturation Lower"]=saturation_lower,
                Para_dict["Saturation Higher"]=saturation_higher,
                Para_dict["Hue on"]=hue_on,
                Para_dict["Hue delta"]=hue_delta,

                Para_dict["Average blur on"]=avgBlur_on,
                Para_dict["Average blur Lower"]=avgBlur_min,
                Para_dict["Average blur Higher"]=avgBlur_max,
                Para_dict["Gauss blur on"]=gaussBlur_on,
                Para_dict["Gauss blur Lower"]=gaussBlur_min,
                Para_dict["Gauss blur Higher"]=gaussBlur_max,
                Para_dict["Motion blur on"]=motionBlur_on,
                Para_dict["Motion blur Kernel"]=motionBlur_kernel,
                Para_dict["Motion blur Angle"]=motionBlur_angle,

                Para_dict["Epoch_Started_Using_These_Settings"]=counter,

                Para_dict["expert_mode"]=expert_mode,
                Para_dict["batchSize_expert"]=batchSize_expert,
                Para_dict["epochs_expert"]=epochs_expert,

                Para_dict["learning_rate_expert_on"]=learning_rate_expert_on,
                Para_dict["learning_rate_const_on"]=learning_rate_const_on,
                Para_dict["learning_rate_const"]=learning_rate_const,
                Para_dict["learning_rate_cycLR_on"]=learning_rate_cycLR_on,
                Para_dict["cycLrMin"]=cycLrMin,
                Para_dict["cycLrMax"]=cycLrMax,
                Para_dict["cycLrMethod"] = cycLrMethod,
                Para_dict["clr_settings"] = self.fittingpopups_ui[listindex].clr_settings,

                Para_dict["learning_rate_expo_on"]=learning_rate_expo_on,
                Para_dict["expDecInitLr"]=expDecInitLr,
                Para_dict["expDecSteps"]=expDecSteps,
                Para_dict["expDecRate"]=expDecRate,

                Para_dict["loss_expert_on"]=loss_expert_on,
                Para_dict["loss_expert"]=loss_expert,
                Para_dict["optimizer_expert_on"]=optimizer_expert_on,
                Para_dict["optimizer_expert"]=optimizer_expert,
                Para_dict["optimizer_settings"]=optimizer_settings,

                Para_dict["paddingMode"]=paddingMode,

                Para_dict["train_last_layers"]=train_last_layers,
                Para_dict["train_last_layers_n"]=train_last_layers_n,
                Para_dict["train_dense_layers"]=train_dense_layers,
                Para_dict["dropout_expert_on"]=dropout_expert_on,
                Para_dict["dropout_expert"]=dropout_expert,
                Para_dict["lossW_expert_on"]=lossW_expert_on,
                Para_dict["lossW_expert"]=lossW_expert,
                Para_dict["class_weight"]=class_weight,
                Para_dict["metrics"]=model_metrics,

                #training data cannot be changed during training
                if norm == "StdScaling using mean and std of all training data":
                    #This needs to be saved into Para_dict since it will be required for inference
                    Para_dict["Mean of training data used for scaling"]=mean_trainingdata,
                    Para_dict["Std of training data used for scaling"]=std_trainingdata,

                if collection==False:
                    if counter == 0:
                        Para_dict.to_excel(self.fittingpopups_ui[listindex].writer,sheet_name='Parameters')
                    else:
                        Para_dict.to_excel(self.fittingpopups_ui[listindex].writer,sheet_name='Parameters',startrow=self.fittingpopups_ui[listindex].writer.sheets['Parameters'].max_row,header= False)

                    if os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                        os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH)#change to read/write
                    try:
                        self.fittingpopups_ui[listindex].writer.save()
                    except:
                        pass
                    os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)#change to only readable

                if collection==True:
                    for i in range(len(Writers)):
                        Para_dict["Chosen Model"]=model_architecture_names[i],
                        writer = Writers[i]
                        if counter==0:
                            Para_dict.to_excel(Writers[i],sheet_name='Parameters')
                        else:
                            Para_dict.to_excel(writer,sheet_name='Parameters',startrow=writer.sheets['Parameters'].max_row,header= False)

                        if os.path.isfile(model_keras_path[i].split(".model")[0]+'_meta.xlsx'):
                            os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #read/write
                        try:
                            writer.save()
                        except:
                            pass
                        os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH) #read only


            ######################Load the Validation Data################################
            ind = [selectedfile["TrainOrValid"] == "Valid" for selectedfile in SelectedFiles]
            ind = np.where(np.array(ind)==True)[0]
            SelectedFiles_valid = np.array(SelectedFiles)[ind]
            SelectedFiles_valid = list(SelectedFiles_valid)
            indices_valid = [selectedfile["class"] for selectedfile in SelectedFiles_valid]
            nr_events_epoch_valid = [selectedfile["nr_events_epoch"] for selectedfile in SelectedFiles_valid]
            rtdc_path_valid = [selectedfile["rtdc_path"] for selectedfile in SelectedFiles_valid]
            zoom_factors_valid = [selectedfile["zoom_factor"] for selectedfile in SelectedFiles_valid]
            #zoom_order = [self.actionOrder0.isChecked(),self.actionOrder1.isChecked(),self.actionOrder2.isChecked(),self.actionOrder3.isChecked(),self.actionOrder4.isChecked(),self.actionOrder5.isChecked()]
            #zoom_order = int(np.where(np.array(zoom_order)==True)[0])
            zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order
            shuffle_valid = [selectedfile["shuffle"] for selectedfile in SelectedFiles_valid]
            xtra_in = {selectedfile["xtra_in"] for selectedfile in SelectedFiles_valid}
            if len(xtra_in)>1:# False and True is present. Not supported
                print("Xtra data is used only for some files. Xtra data needs to be used either by all or by none!")
                return
            xtra_in = list(xtra_in)[0]#this is either True or False

            ############Cropping#####################
            X_valid,y_valid,Indices,xtra_valid = [],[],[],[]

# =============================================================================
#             for i in range(len(SelectedFiles_valid)):
#                 if not self.actionDataToRam.isChecked():
#                     #Replace=true means individual cells could occur several times
#                     gen_valid = aid_img.gen_crop_img(crop,rtdc_path_valid[i],nr_events_epoch_valid[i],random_images=shuffle_valid[i],replace=True,zoom_factor=zoom_factors_valid[i],zoom_order=zoom_order,color_mode=self.get_color_mode(),padding_mode=paddingMode,xtra_in=xtra_in)
#                 else: #get a similar generator, using the ram-data
#                     if len(DATA)==0:
#                         #Replace=true means individual cells could occur several times
#                         gen_valid = aid_img.gen_crop_img(crop,rtdc_path_valid[i],nr_events_epoch_valid[i],random_images=shuffle_valid[i],replace=True,zoom_factor=zoom_factors_valid[i],zoom_order=zoom_order,color_mode=self.get_color_mode(),padding_mode=paddingMode,xtra_in=xtra_in)
#                     else:
#                         gen_valid = aid_img.gen_crop_img_ram(DATA,rtdc_path_valid[i],nr_events_epoch_valid[i],random_images=shuffle_valid[i],replace=True,xtra_in=xtra_in) #Replace true means that individual cells could occur several times
# =============================================================================
            for i in range(len(SelectedFiles_valid)):
                if len(DATA)==0:
                    #Replace=true means individual cells could occur several times
                    gen_valid = aid_img.gen_crop_img(crop,rtdc_path_valid[i],nr_events_epoch_valid[i],random_images=shuffle_valid[i],replace=True,zoom_factor=zoom_factors_valid[i],zoom_order=zoom_order,color_mode=self.get_color_mode(),padding_mode=paddingMode,xtra_in=xtra_in)
                else:
                    gen_valid = aid_img.gen_crop_img_ram(DATA,rtdc_path_valid[i],nr_events_epoch_valid[i],random_images=shuffle_valid[i],replace=True,xtra_in=xtra_in) #Replace true means that individual cells could occur several times


# =============================================================================
#                         if self.actionVerbose.isChecked():
#                             print("Loaded data from RAM")
# =============================================================================
                generator_cropped_out = next(gen_valid)
                X_valid.append(generator_cropped_out[0])
                #y_valid.append(np.repeat(indices_valid[i],nr_events_epoch_valid[i]))
                y_valid.append(np.repeat(indices_valid[i],X_valid[-1].shape[0]))
                Indices.append(generator_cropped_out[1])
                xtra_valid.append(generator_cropped_out[2])
                del generator_cropped_out


# =============================================================================
#             #Save the validation set (BEFORE normalization!)
#             #Write to.rtdc files
#             if bool(self.actionExport_Original.isChecked())==True:
#                 print("Export original images")
#                 save_cropped = False
#                 aid_bin.write_rtdc(new_modelname.split(".model")[0]+'_Valid_Data.rtdc',rtdc_path_valid,X_valid,Indices,cropped=save_cropped,color_mode=self.get_color_mode(),xtra_in=xtra_valid)
#
#             elif bool(self.actionExport_Cropped.isChecked())==True:
#                 print("Export cropped images")
#                 save_cropped = True
#                 aid_bin.write_rtdc(new_modelname.split(".model")[0]+'_Valid_Data.rtdc',rtdc_path_valid,X_valid,Indices,cropped=save_cropped,color_mode=self.get_color_mode(),xtra_in=xtra_valid)
#
#             elif bool(self.actionExport_Off.isChecked())==True:
#                 print("Exporting is turned off")
#     #                msg = QtWidgets.QMessageBox()
#     #                msg.setIcon(QtWidgets.QMessageBox.Information)
#     #                msg.setText("Use a different Exporting option in ->Edit if you want to export the data")
#     #                msg.setWindowTitle("Export is turned off!")
#     #                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
#     #                msg.exec_()
# =============================================================================
            save_cropped = False
            aid_bin.write_rtdc(new_modelname.split(".model")[0]+'_Valid_Data.rtdc',rtdc_path_valid,X_valid,Indices,cropped=save_cropped,color_mode=self.get_color_mode(),xtra_in=xtra_valid)



            X_valid = np.concatenate(X_valid)
            y_valid = np.concatenate(y_valid)
            Y_valid = to_categorical(y_valid, nr_classes)# * 2 - 1
            xtra_valid = np.concatenate(xtra_valid)
# =============================================================================
#             if not bool(self.actionExport_Off.isChecked())==True:
#                 #Save the labels
#                 np.savetxt(new_modelname.split(".model")[0]+'_Valid_Labels.txt',y_valid.astype(int),fmt='%i')
#
# =============================================================================
            #Save the labels
            np.savetxt(new_modelname.split(".model")[0]+'_Valid_Labels.txt',y_valid.astype(int),fmt='%i')


            if len(X_valid.shape)==4:
                channels=3
            elif len(X_valid.shape)==3:
                channels=1
            else:
                print("Invalid data dimension:" +str(X_valid.shape))
            if channels==1:
                #Add the "channels" dimension
                X_valid = np.expand_dims(X_valid,3)

            #get it to theano image format (channels first)
            #X_valid = X_valid.swapaxes(-1,-2).swapaxes(-2,-3)
            if norm == "StdScaling using mean and std of all training data":
                X_valid = aid_img.image_normalization(X_valid,norm,mean_trainingdata,std_trainingdata)
            else:
                X_valid = aid_img.image_normalization(X_valid,norm)

            #Validation data can be cropped to final size already since no augmentation
            #will happen on this data set
            dim_val = X_valid.shape
            print("Current dim. of validation set (pixels x pixels) = "+str(dim_val[2]))
            if dim_val[2]!=crop:
                print("Change dim. (pixels x pixels) of validation set to = "+str(crop))
                remove = int(dim_val[2]/2.0 - crop/2.0)
                X_valid = X_valid[:,remove:remove+crop,remove:remove+crop,:] #crop to crop x crop pixels #TensorFlow

            if xtra_in==True:
                print("Add Xtra Data to X_valid")
                X_valid = [X_valid,xtra_valid]


            ####################Update the PopupFitting########################
            self.fittingpopups_ui[listindex].spinBox_NrEpochs.setValue(nr_epochs)
            chosen_model = str(self.comboBox_ModelSelection.currentText())
            #padding
            index = self.fittingpopups_ui[listindex].comboBox_paddingMode_pop.findText(paddingMode, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.fittingpopups_ui[listindex].comboBox_paddingMode_pop.setCurrentIndex(index)



            index = self.fittingpopups_ui[listindex].comboBox_paddingMode_pop.findText(paddingMode, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.fittingpopups_ui[listindex].comboBox_paddingMode_pop.setCurrentIndex(index)

            do_text = [str(do_i) for do_i in dropout_expert]


            ###############Continue with training data:augmentation############
            #Rotating could create edge effects. Avoid this by making crop a bit larger for now
            #Worst case would be a 45degree rotation:
            cropsize2 = np.sqrt(crop**2+crop**2)
            cropsize2 = np.ceil(cropsize2 / 2.) * 2 #round to the next even number

            #Dictionary defining affine image augmentation options:
            aug_paras = {"v_flip":v_flip,"h_flip":h_flip,"rotation":rotation,"width_shift":width_shift,"height_shift":height_shift,"zoom":zoom,"shear":shear}

            Histories,Index,Saved,Stopwatch,LearningRate = [],[],[],[],[]
            if collection==True:
               HISTORIES = [ [] for model in model_keras]
               SAVED = [ [] for model in model_keras]

            counter = 0
            saving_failed = False #when saving fails, this becomes true and the user will be informed at the end of training

            #Save the initial values (Epoch 1)
            update_para_dict()

            #Dictionary for records in metrics
            model_metrics_records = {}
            model_metrics_records["accuracy"] = 0 #accuracy  starts at zero and approaches 1 during training
            model_metrics_records["val_accuracy"] = 0 #accuracy  starts at zero and approaches 1 during training
            model_metrics_records["loss"] = 9E20 ##loss starts very high and approaches 0 during training
            model_metrics_records["val_loss"] = 9E20 ##loss starts very high and approaches 0 during training
            for key in model_keras.metrics_names:
                if 'precision' in key or 'recall' in key or 'auc' in key:
                    model_metrics_records[key] = 0 #those metrics start at zero and approach 1
                    model_metrics_records["val_"+key] = 0 #those metrics start at zero and approach 1

            gen_train_refresh = False
            time_start = time.time()
            t1 = time.time() #Initialize a timer; this is used to save the meta file every few seconds
            t2 =  time.time() #Initialize a timer; this is used update the fitting parameters
            while counter < nr_epochs:#nr_epochs: #resample nr_epochs times
                #Only keep fitting if the respective window is open:
                isVisible = self.fittingpopups[listindex].isVisible()
                if isVisible:
                    ############Keras image augmentation#####################
                    #Start the first iteration:
                    X_train,y_train,xtra_train = [],[],[]
                    t3 = time.time()
                    for i in range(len(SelectedFiles_train)):
                        if len(DATA)==0 or gen_train_refresh:
                            #Replace true means that individual cells could occur several times
                            gen_train = aid_img.gen_crop_img(cropsize2,rtdc_path_train[i],nr_events_epoch_train[i],random_images=shuffle_train[i],replace=True,zoom_factor=zoom_factors_train[i],zoom_order=zoom_order,color_mode=self.get_color_mode(),padding_mode=paddingMode,xtra_in=xtra_in)
                            gen_train_refresh = False
                        else:
                            gen_train = aid_img.gen_crop_img_ram(DATA,rtdc_path_train[i],nr_events_epoch_train[i],random_images=shuffle_train[i],replace=True,xtra_in=xtra_in) #Replace true means that individual cells could occur several times
# =============================================================================
#                             if self.actionVerbose.isChecked():
#                                 print("Loaded data from RAM")
# =============================================================================
                        data_ = next(gen_train)
                        X_train.append(data_[0])
                        y_train.append(np.repeat(indices_train[i],X_train[-1].shape[0]))
                        if xtra_in==True:
                            xtra_train.append(data_[2])
                        del data_

                    X_train = np.concatenate(X_train)
                    X_train = X_train.astype(np.uint8)
                    y_train = np.concatenate(y_train)
                    if xtra_in==True:
                        print("Retrieve Xtra Data...")
                        xtra_train = np.concatenate(xtra_train)

                    t4 = time.time()
                    if verbose == 1:
                        print("Time to load data (from .rtdc or RAM) and crop="+str(t4-t3))

                    if len(X_train.shape)==4:
                        channels=3
                    elif len(X_train.shape)==3:
                        channels=1
                    else:
                        print("Invalid data dimension:" +str(X_train.shape))
                    if channels==1:
                        #Add the "channels" dimension
                        X_train = np.expand_dims(X_train,3)

                    t3 = time.time()
                    #Some parallellization: use nr_threads (number of CPUs)
                    nr_threads = 1 #Somehow for MNIST and CIFAR, processing always took longer for nr_threads>1 . I tried nr_threads=2,4,8,16,24
                    if nr_threads == 1:
                        X_batch = aid_img.affine_augm(X_train,v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear) #Affine image augmentation
                        y_batch = np.copy(y_train)
                    else:
                        #Divde data in 4 batches
                        X_train = np.array_split(X_train,nr_threads)
                        y_train = np.array_split(y_train,nr_threads)

                        self.X_batch = [False] * nr_threads
                        self.y_batch = [False] * nr_threads
                        self.counter_aug = 0
                        self.Workers_augm = []

                        def imgaug_worker(aug_paras,progress_callback,history_callback):
                            i = aug_paras["i"]
                            self.X_batch[i] = aid_img.affine_augm(aug_paras["X_train"],v_flip,h_flip,rotation,width_shift,height_shift,zoom,shear)
                            self.y_batch[i] = aug_paras["y_train"]
                            self.counter_aug+=1

                        t3_a = time.time()
                        for i in range(nr_threads):
                            aug_paras_ = copy.deepcopy(aug_paras)
                            aug_paras_["i"] = i
                            aug_paras_["X_train"]=X_train[i]#augparas contains rotation and so on. X_train and y_train are overwritten in each iteration (for each worker new X_train)
                            aug_paras_["y_train"]=y_train[i]

                            self.Workers_augm.append(Worker(imgaug_worker,aug_paras_))
                            self.threadpool.start(self.Workers_augm[i])

                        while self.counter_aug < nr_threads:
                            time.sleep(0.01)#Wait 0.1s, then check the len again
                        t3_b = time.time()
                        if verbose == 1:
                            print("Time to perform affine augmentation_internal ="+str(t3_b-t3_a))

                        X_batch = np.concatenate(self.X_batch)
                        y_batch = np.concatenate(self.y_batch)

                    Y_batch = to_categorical(y_batch, nr_classes)# * 2 - 1
                    t4 = time.time()
                    if verbose == 1:
                        print("Time to perform affine augmentation ="+str(t4-t3))

                    t3 = time.time()
                    #Now do the final cropping to the actual size that was set by user
                    dim = X_batch.shape
                    if dim[2]!=crop:
                        remove = int(dim[2]/2.0 - crop/2.0)
                        X_batch = X_batch[:,remove:remove+crop,remove:remove+crop,:] #crop to crop x crop pixels #TensorFlow
                    t4 = time.time()
    #                    if verbose == 1:
    #                        print("Time to crop to final size="+str(t4-t3))

                    X_batch_orig = np.copy(X_batch) #save into new array and do some iterations with varying noise/brightness
                    #reuse this X_batch_orig a few times since this augmentation was costly
                    keras_iter_counter = 0
                    while keras_iter_counter < keras_refresh_nr_epochs and counter < nr_epochs:
                        keras_iter_counter+=1
                        #if t2-t1>5: #check for changed settings every 5 seconds
# =============================================================================
#                         if self.actionVerbose.isChecked()==True:
#                             verbose = 1
#                         else:
# =============================================================================
                        verbose = 0

                        #Another while loop if the user wants to reuse the keras-augmented data
                        #several times and only apply brightness augmentation:
                        brightness_iter_counter = 0
                        while brightness_iter_counter < brightness_refresh_nr_epochs and counter < nr_epochs:
                            #In each iteration, start with non-augmented data
                            X_batch = np.copy(X_batch_orig)#copy from X_batch_orig, X_batch will be altered without altering X_batch_orig
                            X_batch = X_batch.astype(np.uint8)

                            #########X_batch = X_batch.astype(float)########## No float yet :) !!!

                            brightness_iter_counter += 1
# =============================================================================
#                             if self.actionVerbose.isChecked()==True:
#                                 verbose = 1
#                             else:
# =============================================================================
                            verbose = 0

                            if self.fittingpopups_ui[listindex].checkBox_ApplyNextEpoch.isChecked():
                                nr_epochs = int(self.fittingpopups_ui[listindex].spinBox_NrEpochs.value())

                                #brightness_mult_lower = float(self.fittingpopups_ui[listindex].doubleSpinBox_MultLower_pop.value())
                                #gaussnoise_mean = float(self.fittingpopups_ui[listindex].doubleSpinBox_GaussianNoiseMean_pop.value())
                                #gaussnoise_scale = float(self.fittingpopups_ui[listindex].doubleSpinBox_GaussianNoiseScale_pop.value())

                                #contrast_on = bool(self.fittingpopups_ui[listindex].checkBox_contrast_pop.isChecked())
                                #contrast_lower = float(self.fittingpopups_ui[listindex].doubleSpinBox_contrastLower_pop.value())
                                #contrast_higher = float(self.fittingpopups_ui[listindex].doubleSpinBox_contrastHigher_pop.value())
                                #saturation_on = bool(self.fittingpopups_ui[listindex].checkBox_saturation_pop.isChecked())
                                #saturation_lower = float(self.fittingpopups_ui[listindex].doubleSpinBox_saturationLower_pop.value())
                                #saturation_higher = float(self.fittingpopups_ui[listindex].doubleSpinBox_saturationHigher_pop.value())
                                #hue_on = bool(self.fittingpopups_ui[listindex].checkBox_hue_pop.isChecked())
                                #hue_delta = float(self.fittingpopups_ui[listindex].doubleSpinBox_hueDelta_pop.value())




                                #motionBlur_kernel = str(self.fittingpopups_ui[listindex].lineEdit_motionBlurKernel_pop.text())
                                #motionBlur_angle = str(self.fittingpopups_ui[listindex].lineEdit_motionBlurAngle_pop.text())

                                #motionBlur_kernel = tuple(ast.literal_eval(motionBlur_kernel)) #translate string in the lineEdits to a tuple
                                #motionBlur_angle = tuple(ast.literal_eval(motionBlur_angle)) #translate string in the lineEdits to a tuple

                                #Expert mode stuff
                                expert_mode = False
                                #epochs_expert = int(self.fittingpopups_ui[listindex].spinBox_epochs.value())



                                cycLrMin = []
                                cycLrMax = []
                                #cycLrMethod = str(self.fittingpopups_ui[listindex].comboBox_cycLrMethod.currentText())
                                clr_settings = self.fittingpopups_ui[listindex].clr_settings.copy() #Get a copy of the current optimizer_settings. .copy prevents that changes in the UI have immediate effect
                                cycLrStepSize = aid_dl.get_cyclStepSize(SelectedFiles,clr_settings["step_size"],batchSize_expert)
                                cycLrGamma = clr_settings["gamma"]





                                #loss_expert = str(self.fittingpopups_ui[listindex].comboBox_expt_loss_pop.currentText())
                                optimizer_settings = self.fittingpopups_ui[listindex].optimizer_settings.copy() #Get a copy of the current optimizer_settings. .copy prevents that changes in the UI have immediate effect
                                paddingMode_ = str(self.fittingpopups_ui[listindex].comboBox_paddingMode_pop.currentText())
                                print("paddingMode_:"+str(paddingMode_))
                                if paddingMode_ != paddingMode:
                                    print("Changed the padding mode!")
                                    gen_train_refresh = True#otherwise changing paddingMode will not have any effect
                                    paddingMode = paddingMode_

                                try:
                                    dropout_expert = "["+dropout_expert+"]"
                                    dropout_expert = ast.literal_eval(dropout_expert)
                                except:
                                    dropout_expert = []
                                #lossW_expert = str(self.fittingpopups_ui[listindex].lineEdit_lossW.text())
                                class_weight = self.get_class_weight(self.fittingpopups_ui[listindex].SelectedFiles,lossW_expert) #

                                print("Updating parameter file (meta.xlsx)!")
                                update_para_dict()

                                text_updates = ""
                                #Compare current lr and the lr on expert tab:
                                if collection==False:
                                    lr_current = model_keras.optimizer.get_config()["learning_rate"]
                                else:
                                    lr_current = model_keras[0].optimizer.get_config()["learning_rate"]

                                lr_diff = learning_rate_const-lr_current
                                if  abs(lr_diff) > 1e-6:
                                    if collection==False:
                                        K.set_value(model_keras.optimizer.lr, learning_rate_const)
                                    else:
                                        K.set_value(model_keras[0].optimizer.lr, learning_rate_const)

                                    text_updates +=  "Changed the learning rate to "+ str(learning_rate_const)+"\n"

                                recompile = False
                                #Compare current optimizer and the optimizer on expert tab:
                                if collection==False:
                                    optimizer_current = aid_dl.get_optimizer_name(model_keras).lower()#get the current optimizer of the model
                                else:
                                    optimizer_current = aid_dl.get_optimizer_name(model_keras[0]).lower()#get the current optimizer of the model

                                if optimizer_current!=optimizer_expert.lower():#if the current model has a different optimizer
                                    recompile = True
                                    text_updates+="Changed the optimizer to "+optimizer_expert+"\n"

                                #Compare current loss function and the loss-function on expert tab:
                                if collection==False:
                                    loss_ = model_keras.loss
                                else:
                                    loss_ = model_keras[0].loss
                                if loss_!=loss_expert:
                                    recompile = True
                                    model_metrics_records["loss"] = 9E20 #Reset the record for loss because new loss function could converge to a different min. value
                                    model_metrics_records["val_loss"] = 9E20 #Reset the record for loss because new loss function could converge to a different min. value
                                    text_updates+="Changed the loss function to "+loss_expert+"\n"

                                if recompile==True and collection==False:
                                    print("Recompiling...")
                                    model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                                    aid_dl.model_compile(model_keras,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
                                    if model_keras_p!=None:#if model_keras_p is NOT None, there exists a parallel model, which also needs to be re-compiled
                                        model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                                        aid_dl.model_compile(model_keras_p,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)
                                        print("Recompiled parallel model to change optimizer, loss and learninig rate.")

                                elif recompile==True and collection==True:
                                    if model_keras_p!=None:#if model_keras_p is NOT None, there exists a parallel model, which also needs to be re-compiled
                                        print("Altering learning rate is not suported for collections (yet)")
                                        return
                                    print("Recompiling...")
                                    for m in model_keras:
                                        model_metrics_t = aid_dl.get_metrics_tensors(self.get_metrics(),nr_classes)
                                        aid_dl.model_compile(m,loss_expert,optimizer_settings,learning_rate_const,model_metrics_t,nr_classes)

                                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text_updates)

                                #self.model_keras = model_keras #overwrite the model in self
                                self.fittingpopups_ui[listindex].checkBox_ApplyNextEpoch.setChecked(False)


                            ##########Contrast/Saturation/Hue augmentation#########
                            #is there any of contrast/saturation/hue augmentation to do?
                            X_batch = X_batch.astype(np.uint8)
                            if contrast_on:
                                t_con_aug_1 = time.time()
                                X_batch = aid_img.contrast_augm_cv2(X_batch,contrast_lower,contrast_higher) #this function is almost 15 times faster than random_contrast from tf!
                                t_con_aug_2 = time.time()
                                if verbose == 1:
                                    print("Time to augment contrast="+str(t_con_aug_2-t_con_aug_1))

                            if saturation_on or hue_on:
                                t_sat_aug_1 = time.time()
                                X_batch = aid_img.satur_hue_augm_cv2(X_batch.astype(np.uint8),saturation_on,saturation_lower,saturation_higher,hue_on,hue_delta) #Gray and RGB; both values >0!
                                t_sat_aug_2 = time.time()
                                if verbose == 1:
                                    print("Time to augment saturation/hue="+str(t_sat_aug_2-t_sat_aug_1))

                            ##########Average/Gauss/Motion blurring#########
                            #is there any of blurring to do?

                            if avgBlur_on:
                                t_avgBlur_1 = time.time()
                                X_batch = aid_img.avg_blur_cv2(X_batch,avgBlur_min,avgBlur_max)
                                t_avgBlur_2 = time.time()
                                if verbose == 1:
                                    print("Time to perform average blurring="+str(t_avgBlur_2-t_avgBlur_1))

                            if gaussBlur_on:
                                t_gaussBlur_1 = time.time()
                                X_batch = aid_img.gauss_blur_cv(X_batch,gaussBlur_min,gaussBlur_max)
                                t_gaussBlur_2 = time.time()
                                if verbose == 1:
                                    print("Time to perform gaussian blurring="+str(t_gaussBlur_2-t_gaussBlur_1))

                            if motionBlur_on:
                                t_motionBlur_1 = time.time()
                                X_batch = aid_img.motion_blur_cv(X_batch,motionBlur_kernel,motionBlur_angle)
                                t_motionBlur_2 = time.time()
                                if verbose == 1:
                                    print("Time to perform motion blurring="+str(t_motionBlur_2-t_motionBlur_1))

                            ##########Brightness noise#########
                            t3 = time.time()
                            X_batch = aid_img.brightn_noise_augm_cv2(X_batch,brightness_add_lower,brightness_add_upper,brightness_mult_lower,brightness_mult_upper,gaussnoise_mean,gaussnoise_scale)
                            t4 = time.time()
                            if verbose == 1:
                                print("Time to augment brightness="+str(t4-t3))

                            t3 = time.time()
                            if norm == "StdScaling using mean and std of all training data":
                                X_batch = aid_img.image_normalization(X_batch,norm,mean_trainingdata,std_trainingdata)
                            else:
                                X_batch = aid_img.image_normalization(X_batch,norm)
                            t4 = time.time()
                            if verbose == 1:
                                print("Time to apply normalization="+str(t4-t3))

                            #Fitting can be paused
                            while str(self.fittingpopups_ui[listindex].pushButton_Pause_pop.text())==" ":
                                time.sleep(1) #wait 1 seconds and then check the text on the button again

                            if verbose == 1:
                                print("X_batch.shape")
                                print(X_batch.shape)

                            if xtra_in==True:
                                print("Add Xtra Data to X_batch")
                                X_batch = [X_batch,xtra_train]

                            #generate a list of callbacks, get empty list if callback_lr is none
                            callbacks = []
                            if callback_lr!=None:
                                callbacks.append(callback_lr)

                            ###################################################
                            ###############Actual fitting######################
                            ###################################################
                            if collection==False:
                                if model_keras_p == None:
                                    history = model_keras.fit(X_batch, Y_batch, batch_size=batchSize_expert,epochs=epochs_expert,verbose=verbose,\
                                                              validation_data=(X_valid, Y_valid),class_weight=class_weight,callbacks=callbacks)
                                elif model_keras_p != None:
                                    history = model_keras_p.fit(X_batch, Y_batch, batch_size=batchSize_expert, epochs=epochs_expert,verbose=verbose,\
                                                                validation_data=(X_valid, Y_valid),class_weight=class_weight,callbacks=callbacks)

                                Histories.append(history.history)
                                Stopwatch.append(time.time()-time_start)
                                learningrate = K.get_value(history.model.optimizer.lr)
                                LearningRate.append(learningrate)

                                #Check if any metric broke a record
                                record_broken = False #initially, assume there is no new record
                                for key in history.history.keys():
                                    value = history.history[key][-1]
                                    record = model_metrics_records[key]
                                    if 'val_accuracy' in key or 'val_precision' in key or 'val_recall' in key or 'val_auc' in key:
                                        #These metrics should go up (towards 1)
                                        if value>record:
                                            model_metrics_records[key] = value
                                            record_broken = True
                                            print(key+" broke record -> Model will be saved" )

                                    elif 'val_loss' in key:
                                        #This metric should go down (towards 0)
                                        if value<record:
                                            model_metrics_records[key] = value
                                            record_broken = True
                                            print(key+" broke record -> Model will be saved")
                                                #self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                if record_broken:#if any record was broken...
# =============================================================================
#                                     if deviceSelected=="Multi-GPU":#in case of Multi-GPU...
#                                         #In case of multi-GPU, first copy the weights of the parallel model to the normal model
#                                         model_keras.set_weights(model_keras_p.layers[-2].get_weights())
# =============================================================================
                                    #Save the model
                                    text = "Save model to following directory: \n"+os.path.dirname(new_modelname)
                                    print(text)

                                    if os.path.exists(os.path.dirname(new_modelname)):
                                        model_keras.save(new_modelname.split(".model")[0]+"_"+str(counter)+".model",save_format='h5')
                                        text = "Record was broken -> saved model"
                                        print(text)
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                    else:#in case the folder does not exist (anymore), create a folder in temp
                                        #what is the foldername of the model?
                                        text = "Saving failed. Create folder in temp"
                                        print(text)
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                        saving_failed = True
                                        temp_path = aid_bin.create_temp_folder()#create a temp folder if it does not already exist

                                        text = "Your temp. folder is here: "+str(temp_path)
                                        print(text)
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                        parentfolder = aid_bin.splitall(new_modelname)[-2]
                                        fname = os.path.split(new_modelname)[-1]

                                        #create that folder in temp if it not exists already
                                        if not os.path.exists(os.path.join(temp_path,parentfolder)):
                                            text = "Create folder in temp:\n"+os.path.join(temp_path,parentfolder)
                                            print(text)
                                            self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)
                                            os.mkdir(os.path.join(temp_path,parentfolder))

                                        #change the new_modelname to a path in temp
                                        new_modelname = os.path.join(temp_path,parentfolder,fname)

                                        #inform user!
                                        text = "Could not find original folder. Files are now saved to "+new_modelname
                                        text = "<span style=\' color: red;\'>" +text+"</span>"
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)
                                        text = "<span style=\' color: black;\'>" +""+"</span>"
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                        #Save the  model
                                        model_keras.save(new_modelname.split(".model")[0]+"_"+str(counter)+".model",save_format='h5')
                                        text = "Model saved successfully to temp"
                                        print(text)
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                        #Also update the excel writer!
                                        writer = pd.ExcelWriter(new_modelname.split(".model")[0]+'_meta.xlsx', engine='openpyxl')
                                        self.fittingpopups_ui[listindex].writer = writer
                                        pd.DataFrame().to_excel(writer,sheet_name='UsedData') #initialize empty Sheet
                                        SelectedFiles_df.to_excel(writer,sheet_name='UsedData')
                                        DataOverview_df.to_excel(writer,sheet_name='DataOverview') #write data overview to separate sheet
                                        pd.DataFrame().to_excel(writer,sheet_name='Parameters') #initialize empty Sheet
                                        pd.DataFrame().to_excel(writer,sheet_name='History') #initialize empty Sheet

                                    Saved.append(1)

                                #Also save the model upon user-request
                                elif bool(self.fittingpopups_ui[listindex].checkBox_saveEpoch_pop.isChecked())==True:
                                    if deviceSelected=="Multi-GPU":#in case of Multi-GPU...
                                        #In case of multi-GPU, first copy the weights of the parallel model to the normal model
                                        model_keras.set_weights(model_keras_p.layers[-2].get_weights())
                                    model_keras.save(new_modelname.split(".model")[0]+"_"+str(counter)+".model",save_format='h5')
                                    Saved.append(1)
                                    self.fittingpopups_ui[listindex].checkBox_saveEpoch_pop.setChecked(False)
                                else:
                                    Saved.append(0)

                            elif collection==True:
                                for i in range(len(model_keras)):
                                    #Expert-settings return automatically to default values when Expert-mode is unchecked
                                    history = model_keras[i].fit(X_batch, Y_batch, batch_size=batchSize_expert, epochs=epochs_expert,verbose=verbose, validation_data=(X_valid, Y_valid),class_weight=class_weight,callbacks=callbacks)
                                    HISTORIES[i].append(history.history)
                                    learningrate = K.get_value(history.model.optimizer.lr)

                                    print("model_keras_path[i]")
                                    print(model_keras_path[i])

                                    #Check if any metric broke a record
                                    record_broken = False #initially, assume there is no new record

                                    for key in history.history.keys():
                                        value = history.history[key][-1]
                                        record = model_metrics_records[key]
                                        if 'val_accuracy' in key or 'val_precision' in key or 'val_recall' in key or 'val_auc' in key:
                                            #These metrics should go up (towards 1)
                                            if value>record:
                                                model_metrics_records[key] = value
                                                record_broken = True
                                                text = key+" broke record -> Model will be saved"
                                                print(text)
                                                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                                #one could 'break' here, but I want to update all records
                                        elif 'val_loss' in key:
                                            #This metric should go down (towards 0)
                                            if value<record:
                                                model_metrics_records[key] = value
                                                record_broken = True
                                                text = key+" broke record -> Model will be saved"
                                                print(text)
                                                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                    #For collections of models:
                                    if record_broken:
                                        #Save the model
                                        model_keras[i].save(model_keras_path[i].split(".model")[0]+"_"+str(counter)+".model")
                                        SAVED[i].append(1)
                                    elif bool(self.fittingpopups_ui[listindex].checkBox_saveEpoch_pop.isChecked())==True:
                                        model_keras[i].save(model_keras_path[i].split(".model")[0]+"_"+str(counter)+".model")
                                        SAVED[i].append(1)
                                        self.fittingpopups_ui[listindex].checkBox_saveEpoch_pop.setChecked(False)
                                    else:
                                        SAVED[i].append(0)


                            callback_progessbar = float(counter)/nr_epochs
                            progress_callback.emit(100.0*callback_progessbar)
                            history_emit = history.history
                            history_emit["LearningRate"] = [learningrate]
                            history_callback.emit(history_emit)
                            Index.append(counter)

                            t2 =  time.time()

                            if collection==False:
                                if counter==0:
                                    #If this runs the first time, create the file with header
                                    DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                                    DF1 = np.r_[DF1]
                                    DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )

                                    DF1["Saved"] = Saved
                                    DF1["Time"] = Stopwatch
                                    DF1["LearningRate"] = LearningRate
                                    DF1.index = Index

                                    #If this runs the first time, create the file with header
                                    if os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                                        os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #read/write
                                    DF1.to_excel(writer,sheet_name='History')
                                    writer.save()
                                    os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)

                                    meta_saving_t = int(self.fittingpopups_ui[listindex].spinBox_saveMetaEvery.value())
                                    text = "meta.xlsx was saved (automatic saving every "+str(meta_saving_t)+"s)"
                                    print(text)
                                    self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                    #self.fittingpopups_ui[listindex].backup.append({"DF1":DF1})
                                    Index,Histories,Saved,Stopwatch,LearningRate = [],[],[],[],[]#reset the lists

                                #Get a sensible frequency for saving the dataframe (every 20s)
                                elif t2-t1>int(self.fittingpopups_ui[listindex].spinBox_saveMetaEvery.value()):
                                #elif counter%50==0:  #otherwise save the history to excel after each n epochs
                                    DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                                    DF1 = np.r_[DF1]
                                    DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
                                    DF1["Saved"] = Saved
                                    DF1["Time"] = Stopwatch
                                    DF1["LearningRate"] = LearningRate
                                    DF1.index = Index

                                    #Saving
                                    if os.path.exists(os.path.dirname(new_modelname)):#check if folder is (still) available
                                        if os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                                            os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #make read/write
                                        DF1.to_excel(writer,sheet_name='History', startrow=writer.sheets['History'].max_row,header= False)
                                        writer.save()
                                        os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)  #make read only

                                        meta_saving_t = int(self.fittingpopups_ui[listindex].spinBox_saveMetaEvery.value())
                                        text = "meta.xlsx was saved (automatic saving every "+str(meta_saving_t)+"s to directory:\n)"+new_modelname
                                        print(text)
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

                                        Index,Histories,Saved,Stopwatch,LearningRate = [],[],[],[],[]#reset the lists
                                        t1 = time.time()
                                    else:#If folder not available, create a folder in temp
                                        text = "Failed to save meta.xlsx. -> Create folder in temp\n"
                                        saving_failed = True
                                        temp_path = aid_bin.create_temp_folder()#create a temp folder if it does not already exist
                                        text += "Your temp folder is here: "+str(temp_path)+"\n"
                                        folder = os.path.split(new_modelname)[-2]
                                        folder = os.path.split(folder)[-1]
                                        fname = os.path.split(new_modelname)[-1]
                                        #create that folder in temp if it does'nt exist already
                                        if not os.path.exists(os.path.join(temp_path,folder)):
                                            os.mkdir(os.path.join(temp_path,folder))
                                            text +="Created directory in temp:\n"+os.path.join(temp_path,folder)

                                        print(text)
                                        #change the new_modelname to a path in temp
                                        new_modelname = os.path.join(temp_path,folder,fname)

                                        #inform user!
                                        text = "Could not find original folder. Files are now saved to "+new_modelname
                                        text = "<span style=\' color: red;\'>" +text+"</span>"#put red text to the infobox
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)
                                        text = "<span style=\' color: black;\'>" +""+"</span>"#reset textcolor to black
                                        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)


                                        #update the excel writer
                                        writer = pd.ExcelWriter(new_modelname.split(".model")[0]+'_meta.xlsx', engine='openpyxl')
                                        self.fittingpopups_ui[listindex].writer = writer
                                        pd.DataFrame().to_excel(writer,sheet_name='UsedData') #initialize empty Sheet
                                        SelectedFiles_df.to_excel(writer,sheet_name='UsedData')
                                        DataOverview_df.to_excel(writer,sheet_name='DataOverview') #write data overview to separate sheet
                                        pd.DataFrame().to_excel(writer,sheet_name='Parameters') #initialize empty Sheet
                                        pd.DataFrame().to_excel(writer,sheet_name='History') #initialize empty Sheet

                                        if os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                                            print("There is already such a file...AID will add new data to it. Please check if this is OK")
                                            os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #read/write
                                        DF1.to_excel(writer,sheet_name='History')
                                        writer.save()
                                        os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)
                                        print("meta.xlsx was saved")
                                        Index,Histories,Saved,Stopwatch,LearningRate = [],[],[],[],[]#reset the lists


                            if collection==True:
                                if counter==0:
                                    for i in range(len(HISTORIES)):
                                        Histories = HISTORIES[i]
                                        Saved = SAVED[i]
                                        #If this runs the first time, create the file with header
                                        DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                                        DF1 = np.r_[DF1]
                                        DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
                                        DF1["Saved"] = Saved
                                        DF1.index = Index
                                        HISTORIES[i] = []#reset the Histories list
                                        SAVED[i] = []
                                        #If this runs the first time, create the file with header
                                        if os.path.isfile(model_keras_path[i].split(".model")[0]+'_meta.xlsx'):
                                            os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #read/write
                                        DF1.to_excel(Writers[i],sheet_name='History')
                                        Writers[i].save()
                                        os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)
                                        print("meta.xlsx was saved")
                                    Index = []#reset the Index list

                                #Get a sensible frequency for saving the dataframe (every 20s)
                                elif t2-t1>int(self.fittingpopups_ui[listindex].spinBox_saveMetaEvery.value()):
                                    for i in range(len(HISTORIES)):
                                        Histories = HISTORIES[i]
                                        Saved = SAVED[i]
                                        DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                                        DF1 = np.r_[DF1]
                                        DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
                                        DF1["Saved"] = Saved
                                        DF1.index = Index
                                        HISTORIES[i] = []#reset the Histories list
                                        SAVED[i] = []
                                        #Saving
                                        #TODO: save to temp, if harddisk not available to prevent crash.
                                        if os.path.isfile(model_keras_path[i].split(".model")[0]+'_meta.xlsx'):
                                            os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #make read/write
                                        DF1.to_excel(Writers[i],sheet_name='History', startrow=Writers[i].sheets['History'].max_row,header= False)
                                        Writers[i].save()
                                        os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH)  #make read only
                                        print("meta.xlsx was saved")
                                        t1 = time.time()
                                    Index = []#reset the Index list

                            counter+=1

            progress_callback.emit(100.0)

            #If the original storing locating became inaccessible (folder name changed, HD unplugged...)
            #the models and meta are saved to temp folder. Inform the user!!!
            if saving_failed==True:
# =============================================================================
#                 text = "<html><head/><body><p>Original path:<br>"+path_orig+\
#                 "<br>became inaccessible during training! Files were then saved to:<br>"+\
#                 new_modelname.split(".model")[0]+"<br>To bring both parts back together\
#                 , you have manually open the meta files (excel) and copy;paste each sheet. \
#                 Sorry for the inconvenience.<br>If that happens often, you may contact \
#                 the main developer and ask him to improve that.</p></body></html>"
# =============================================================================

                text = "<span style=\' font-weight:600; color: red;\'>" +text+"</span>"#put red text to the infobox
                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)
                print('\a')#make a noise
                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.setStyleSheet("background-color: yellow;")
                self.fittingpopups_ui[listindex].textBrowser_FittingInfo.moveCursor(QtGui.QTextCursor.End)

            if collection==False:
                if len(Histories)>0: #if the list for History files is not empty, process it!
                    DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                    DF1 = np.r_[DF1]
                    DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
                    DF1["Saved"] = Saved
                    DF1["Time"] = Stopwatch
                    DF1["LearningRate"] = LearningRate
                    DF1.index = Index
                    Index = []#reset the Index list
                    Histories = []#reset the Histories list
                    Saved = []
                    #does such a file exist already? append!
                    if not os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                       DF1.to_excel(writer,sheet_name='History')
                    else: # else it exists so append without writing the header
                       DF1.to_excel(writer,sheet_name='History', startrow=writer.sheets['History'].max_row,header= False)
                if os.path.isfile(new_modelname.split(".model")[0]+'_meta.xlsx'):
                    os.chmod(new_modelname.split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #make read/write
                writer.save()
                writer.close()

            if collection==True:
                for i in range(len(HISTORIES)):
                    Histories = HISTORIES[i]
                    Saved = SAVED[i]
                    if len(Histories)>0: #if the list for History files is not empty, process it!
                        DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
                        DF1 = np.r_[DF1]
                        DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
                        DF1["Saved"] = Saved
                        DF1.index = Index
                        HISTORIES[i] = []#reset the Histories list
                        SAVED[i] = []
                        #does such a file exist already? append!
                        if not os.path.isfile(model_keras_path[i].split(".model")[0]+'_meta.xlsx'):
                           DF1.to_excel(Writers[i],sheet_name='History')
                        else: # else it exists so append without writing the header
                           DF1.to_excel(writer,sheet_name='History', startrow=writer.sheets['History'].max_row,header= False)
                    if os.path.isfile(model_keras_path[i].split(".model")[0]+'_meta.xlsx'):
                        os.chmod(model_keras_path[i].split(".model")[0]+'_meta.xlsx', S_IREAD|S_IRGRP|S_IROTH|S_IWRITE|S_IWGRP|S_IWOTH) #make read/write
                    Writers[i].save()
                    Writers[i].close()

                Index = []#reset the Index list



            sess.close()
    #        try:
    #            aid_dl.reset_keras(model_keras)
    #        except:
    #            pass


    def update_historyplot_pop(self,listindex):
        #listindex = self.popupcounter-1 #len(self.fittingpopups_ui)-1
        #After the first epoch there are checkboxes available. Check, if user checked some:
        colcount = int(self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.columnCount())
        #Collect items that are checked
        selected_items,Colors = [],[]
        for colposition in range(colcount):
            #is it checked for train?
            cb = self.fittingpopups_ui[listindex].tableWidget_HistoryInfo_pop.item(0, colposition)
            if not cb==None:
                if cb.checkState() == QtCore.Qt.Checked:
                    selected_items.append(str(cb.text()))
                    Colors.append(cb.background())
        self.Colors = Colors
        Histories = self.fittingpopups_ui[listindex].Histories
        DF1 = [[ h[h_i][-1] for h_i in h] for h in Histories] #if nb_epoch in .fit() is >1, only save the last history item, beacuse this would a model that could be saved
        DF1 = np.r_[DF1]
        DF1 = pd.DataFrame( DF1,columns=Histories[0].keys() )
        self.fittingpopups_ui[listindex].widget_pop.clear()

        #Create fresh plot
        plt1 = self.fittingpopups_ui[listindex].widget_pop.addPlot()
        plt1.showGrid(x=True,y=True)
        plt1.addLegend()
        plt1.setLabel('bottom', 'Epoch', units='')
        #Create a dict that stores plots for each metric (for real time plotting)
        self.fittingpopups_ui[listindex].historyscatters = dict()
        for i in range(len(selected_items)):
            key = selected_items[i]
            df = DF1[key]
            color = self.Colors[i]
            pen_rollmedi = list(color.color().getRgb())
            pen_rollmedi = pg.mkColor(pen_rollmedi)
            pen_rollmedi = pg.mkPen(color=pen_rollmedi,width=6)
            color = list(color.color().getRgb())
            color[-1] = int(0.6*color[-1])
            color = tuple(color)
            pencolor = pg.mkColor(color)
            brush = pg.mkBrush(color=pencolor)

            historyscatter = plt1.plot(range(len(df)), df.values, pen=None,symbol='o',symbolPen=None,symbolBrush=brush,name=key,clear=False)
            self.fittingpopups_ui[listindex].historyscatters[key]=historyscatter


    def actionDataToRamNow_function(self):
# =============================================================================
#         self.statusbar.showMessage("Moving data to RAM")
# =============================================================================
        #check that the nr. of classes are equal to the model out put
        SelectedFiles = self.items_clicked()
        color_mode = self.get_color_mode()
        zoom_factors = [selectedfile["zoom_factor"] for selectedfile in SelectedFiles]
        #zoom_order = [self.actionOrder0.isChecked(),self.actionOrder1.isChecked(),self.actionOrder2.isChecked(),self.actionOrder3.isChecked(),self.actionOrder4.isChecked(),self.actionOrder5.isChecked()]
        #zoom_order = int(np.where(np.array(zoom_order)==True)[0])
        zoom_order = int(self.comboBox_zoomOrder.currentIndex()) #the combobox-index is already the zoom order

        #Get the user-defined cropping size
        crop = int(self.spinBox_imagecrop.value())
        #Make the cropsize a bit larger since the images will later be rotated
        cropsize2 = np.sqrt(crop**2+crop**2)
        cropsize2 = np.ceil(cropsize2 / 2.) * 2 #round to the next even number

        dic = aid_img.crop_imgs_to_ram(list(SelectedFiles),crop,zoom_factors=zoom_factors,zoom_order=zoom_order,color_mode=color_mode)
        self.ram = dic

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Successfully moved data to RAM")
        msg.setWindowTitle("Moved Data to RAM")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

# =============================================================================
#         self.statusbar.showMessage("")
# =============================================================================

    def get_class_weight(self,SelectedFiles,lossW_expert,custom_check_classes=False):
        t1 = time.time()
        print("Getting dictionary for class_weight")
        if lossW_expert=="None":
            return None
        elif lossW_expert=="":
            return None
        elif lossW_expert=="Balanced":
            #Which are training files?
            ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
            ind = np.where(np.array(ind)==True)[0]
            SelectedFiles_train = list(np.array(SelectedFiles)[ind])
            classes = [int(selectedfile["class"]) for selectedfile in SelectedFiles_train]
            nr_events_epoch = [int(selectedfile["nr_events_epoch"]) for selectedfile in SelectedFiles_train]
            classes_uni = np.unique(classes)
            counter = {}
            for class_ in classes_uni:
                ind = np.where(np.array(classes)==class_)[0]
                nr_events_epoch_class = np.array(nr_events_epoch)[ind]
                counter[class_] = np.sum(nr_events_epoch_class)
            max_val = float(max(counter.values()))
            return {class_id : max_val/num_images for class_id, num_images in counter.items()}

        elif lossW_expert.startswith("{"):#Custom loss weights
            class_weights = eval(lossW_expert)
            if custom_check_classes:#Check that each element in classes_uni is contained in class_weights.keys()
                ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
                ind = np.where(np.array(ind)==True)[0]
                SelectedFiles_train = list(np.array(SelectedFiles)[ind])
                classes = [int(selectedfile["class"]) for selectedfile in SelectedFiles_train]
                classes_uni = np.unique(classes)
                classes_uni = np.sort(classes_uni)
                class_weights_keys = np.sort([int(a) for a in class_weights.keys()])
                #each element in classes_uni has to be equal to class_weights_keys
                equal = np.array_equal(classes_uni,class_weights_keys)
                if equal == True:
                    return class_weights
                else:
                    #If the equal is false I'm really in trouble...
                    #run the function again, but request 'Balanced' weights. I'm not sure if this should be the default...
                    class_weights = self.get_class_weight(SelectedFiles,"Balanced")
                    return ["Balanced",class_weights]
            else:
                return class_weights
        t2 = time.time()
        dt = np.round(t2-t1,2)
        print("Comp. time = "+str(dt))


    def get_dataOverview(self):
        table = self.tableWidget_Info
        cols = table.columnCount()
        header = [table.horizontalHeaderItem(col).text() for col in range(cols)]
        rows = table.rowCount()
        tmp_df = pd.DataFrame(columns=header,index=range(rows))
        for i in range(rows):
            for j in range(cols):
                try:
                    tmp_df.iloc[i, j] = table.item(i, j).text()
                except:
                    tmp_df.iloc[i, j] = np.nan
        return tmp_df

# =============================================================================
# Funtions of pop up window
# =============================================================================
    def pause_fitting_pop(self,listindex):
        #Just change the text on the button
        if str(self.fittingpopups_ui[listindex].pushButton_Pause_pop.text())=="":
            #If the the text on the button was Pause, change it to Continue
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setText(" ")
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setStyleSheet("background-color: green")
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setIcon(QtGui.QIcon(os.path.join(dir_root, "art","Icon","continue.png")))

        elif str(self.fittingpopups_ui[listindex].pushButton_Pause_pop.text())==" ":
            #If the the text on the button was Continue, change it to Pause
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setText("")
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setIcon(QtGui.QIcon(os.path.join(dir_root, "art","Icon","pause.png")))
            self.fittingpopups_ui[listindex].pushButton_Pause_pop.setStyleSheet("")


    def stop_fitting_pop(self,listindex):
        #listindex = len(self.fittingpopups_ui)-1
        epochs = self.fittingpopups_ui[listindex].epoch_counter
        #Stop button on the fititng popup
        #Should stop the fitting process and save the metafile
        #1. Change the nr. requested epochs to a smaller number
        self.fittingpopups_ui[listindex].spinBox_NrEpochs.setValue(epochs-1)
        #2. Check the box which will cause that the new parameters are applied at next epoch
        self.fittingpopups_ui[listindex].checkBox_ApplyNextEpoch.setChecked(True)

    def clearTextWindow_pop(self,listindex):
        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.clear()

    def showModelSumm_pop(self,listindex):
        text5 = "Model summary:\n"
        summary = []
        self.model_keras.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        text = text5+summary
        self.fittingpopups_ui[listindex].textBrowser_FittingInfo.append(text)

    def saveModelSumm_pop(self,listindex):
        text5 = "Model summary:\n"
        summary = []
        self.model_keras.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        text = text5+summary
        #Ask the user where to save the stuff
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Model summary', Default_dict["Path of last model"]," (*.txt)")
        filename = filename[0]
        #Save to this filename
        f = open(filename,'w')
        f.write(text)
        f.close()

    def saveTextWindow_pop(self,listindex):
        #Get the entire content of textBrowser_FittingInfo
        text = str(self.fittingpopups_ui[listindex].textBrowser_FittingInfo.toPlainText())
        #Ask the user where to save the stuff
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Fitting info', Default_dict["Path of last model"]," (*.txt)")
        filename = filename[0]
        #Save to this filename
        if len(filename)>0:
            f = open(filename,'w')
            f.write(text)
            f.close()




