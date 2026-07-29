# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.11.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QSizePolicy, QSpacerItem, QSplitter,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1280, 743)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setSpacing(0)
        self.centralLayout.setObjectName(u"centralLayout")
        self.centralLayout.setContentsMargins(0, 0, 0, 0)
        self.mainSplitter = QSplitter(self.centralwidget)
        self.mainSplitter.setObjectName(u"mainSplitter")
        self.mainSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.mainSplitter.setChildrenCollapsible(False)
        self.sidebarFrame = QFrame(self.mainSplitter)
        self.sidebarFrame.setObjectName(u"sidebarFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sidebarFrame.sizePolicy().hasHeightForWidth())
        self.sidebarFrame.setSizePolicy(sizePolicy)
        self.sidebarFrame.setMinimumSize(QSize(180, 0))
        self.sidebarFrame.setMaximumSize(QSize(400, 16777215))
        self.sidebarLayout = QVBoxLayout(self.sidebarFrame)
        self.sidebarLayout.setSpacing(8)
        self.sidebarLayout.setObjectName(u"sidebarLayout")
        self.sidebarLayout.setContentsMargins(10, 10, 10, 10)
        self.profileFrame = QFrame(self.sidebarFrame)
        self.profileFrame.setObjectName(u"profileFrame")
        self.hboxLayout = QHBoxLayout(self.profileFrame)
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.driverIcon = QLabel(self.profileFrame)
        self.driverIcon.setObjectName(u"driverIcon")

        self.hboxLayout.addWidget(self.driverIcon)

        self.driverName = QLabel(self.profileFrame)
        self.driverName.setObjectName(u"driverName")

        self.hboxLayout.addWidget(self.driverName)


        self.sidebarLayout.addWidget(self.profileFrame)

        self.distractionFrame = QFrame(self.sidebarFrame)
        self.distractionFrame.setObjectName(u"distractionFrame")
        self.vboxLayout = QVBoxLayout(self.distractionFrame)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.lblDistractionVal = QLabel(self.distractionFrame)
        self.lblDistractionVal.setObjectName(u"lblDistractionVal")
        self.lblDistractionVal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vboxLayout.addWidget(self.lblDistractionVal)

        self.lblDistractionTitle = QLabel(self.distractionFrame)
        self.lblDistractionTitle.setObjectName(u"lblDistractionTitle")
        self.lblDistractionTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vboxLayout.addWidget(self.lblDistractionTitle)


        self.sidebarLayout.addWidget(self.distractionFrame)

        self.drowsyFrame = QFrame(self.sidebarFrame)
        self.drowsyFrame.setObjectName(u"drowsyFrame")
        self.vboxLayout1 = QVBoxLayout(self.drowsyFrame)
        self.vboxLayout1.setObjectName(u"vboxLayout1")
        self.lblDrowsyVal = QLabel(self.drowsyFrame)
        self.lblDrowsyVal.setObjectName(u"lblDrowsyVal")
        self.lblDrowsyVal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vboxLayout1.addWidget(self.lblDrowsyVal)

        self.lblDrowsyTitle = QLabel(self.drowsyFrame)
        self.lblDrowsyTitle.setObjectName(u"lblDrowsyTitle")
        self.lblDrowsyTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vboxLayout1.addWidget(self.lblDrowsyTitle)


        self.sidebarLayout.addWidget(self.drowsyFrame)

        self.actionFrame = QFrame(self.sidebarFrame)
        self.actionFrame.setObjectName(u"actionFrame")
        self.vboxLayout2 = QVBoxLayout(self.actionFrame)
        self.vboxLayout2.setObjectName(u"vboxLayout2")
        self.lblActionTitle = QLabel(self.actionFrame)
        self.lblActionTitle.setObjectName(u"lblActionTitle")

        self.vboxLayout2.addWidget(self.lblActionTitle)

        self.lblActionIcons = QLabel(self.actionFrame)
        self.lblActionIcons.setObjectName(u"lblActionIcons")

        self.vboxLayout2.addWidget(self.lblActionIcons)


        self.sidebarLayout.addWidget(self.actionFrame)

        self.expressionFrame = QFrame(self.sidebarFrame)
        self.expressionFrame.setObjectName(u"expressionFrame")
        self.vboxLayout3 = QVBoxLayout(self.expressionFrame)
        self.vboxLayout3.setObjectName(u"vboxLayout3")
        self.lblExpTitle = QLabel(self.expressionFrame)
        self.lblExpTitle.setObjectName(u"lblExpTitle")

        self.vboxLayout3.addWidget(self.lblExpTitle)

        self.lblExpVal = QLabel(self.expressionFrame)
        self.lblExpVal.setObjectName(u"lblExpVal")

        self.vboxLayout3.addWidget(self.lblExpVal)


        self.sidebarLayout.addWidget(self.expressionFrame)

        self.eyeOpFrame = QFrame(self.sidebarFrame)
        self.eyeOpFrame.setObjectName(u"eyeOpFrame")
        self.vboxLayout4 = QVBoxLayout(self.eyeOpFrame)
        self.vboxLayout4.setObjectName(u"vboxLayout4")
        self.lblEyeOpTitle = QLabel(self.eyeOpFrame)
        self.lblEyeOpTitle.setObjectName(u"lblEyeOpTitle")

        self.vboxLayout4.addWidget(self.lblEyeOpTitle)

        self.hboxLayout1 = QHBoxLayout()
        self.hboxLayout1.setObjectName(u"hboxLayout1")
        self.lblEyeOpL = QLabel(self.eyeOpFrame)
        self.lblEyeOpL.setObjectName(u"lblEyeOpL")

        self.hboxLayout1.addWidget(self.lblEyeOpL)

        self.lblEyeOpR = QLabel(self.eyeOpFrame)
        self.lblEyeOpR.setObjectName(u"lblEyeOpR")

        self.hboxLayout1.addWidget(self.lblEyeOpR)


        self.vboxLayout4.addLayout(self.hboxLayout1)


        self.sidebarLayout.addWidget(self.eyeOpFrame)

        self.blinkFrame = QFrame(self.sidebarFrame)
        self.blinkFrame.setObjectName(u"blinkFrame")
        self.vboxLayout5 = QVBoxLayout(self.blinkFrame)
        self.vboxLayout5.setObjectName(u"vboxLayout5")
        self.lblBlinkTitle = QLabel(self.blinkFrame)
        self.lblBlinkTitle.setObjectName(u"lblBlinkTitle")

        self.vboxLayout5.addWidget(self.lblBlinkTitle)

        self.lblBlinkVal = QLabel(self.blinkFrame)
        self.lblBlinkVal.setObjectName(u"lblBlinkVal")

        self.vboxLayout5.addWidget(self.lblBlinkVal)


        self.sidebarLayout.addWidget(self.blinkFrame)

        self.headLocFrame = QFrame(self.sidebarFrame)
        self.headLocFrame.setObjectName(u"headLocFrame")
        self.vboxLayout6 = QVBoxLayout(self.headLocFrame)
        self.vboxLayout6.setObjectName(u"vboxLayout6")
        self.lblHeadLocTitle = QLabel(self.headLocFrame)
        self.lblHeadLocTitle.setObjectName(u"lblHeadLocTitle")

        self.vboxLayout6.addWidget(self.lblHeadLocTitle)

        self.lblHeadLocVal = QLabel(self.headLocFrame)
        self.lblHeadLocVal.setObjectName(u"lblHeadLocVal")

        self.vboxLayout6.addWidget(self.lblHeadLocVal)


        self.sidebarLayout.addWidget(self.headLocFrame)

        self.eyeLocFrame = QFrame(self.sidebarFrame)
        self.eyeLocFrame.setObjectName(u"eyeLocFrame")
        self.vboxLayout7 = QVBoxLayout(self.eyeLocFrame)
        self.vboxLayout7.setObjectName(u"vboxLayout7")
        self.lblEyeLocTitle = QLabel(self.eyeLocFrame)
        self.lblEyeLocTitle.setObjectName(u"lblEyeLocTitle")

        self.vboxLayout7.addWidget(self.lblEyeLocTitle)

        self.lblEyeLocVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocVal.setObjectName(u"lblEyeLocVal")

        self.vboxLayout7.addWidget(self.lblEyeLocVal)


        self.sidebarLayout.addWidget(self.eyeLocFrame)

        self.headDirFrame = QFrame(self.sidebarFrame)
        self.headDirFrame.setObjectName(u"headDirFrame")
        self.vboxLayout8 = QVBoxLayout(self.headDirFrame)
        self.vboxLayout8.setObjectName(u"vboxLayout8")
        self.lblHeadDirTitle = QLabel(self.headDirFrame)
        self.lblHeadDirTitle.setObjectName(u"lblHeadDirTitle")

        self.vboxLayout8.addWidget(self.lblHeadDirTitle)

        self.lblHeadDirVal = QLabel(self.headDirFrame)
        self.lblHeadDirVal.setObjectName(u"lblHeadDirVal")

        self.vboxLayout8.addWidget(self.lblHeadDirVal)


        self.sidebarLayout.addWidget(self.headDirFrame)

        self.gazeDirFrame = QFrame(self.sidebarFrame)
        self.gazeDirFrame.setObjectName(u"gazeDirFrame")
        self.vboxLayout9 = QVBoxLayout(self.gazeDirFrame)
        self.vboxLayout9.setObjectName(u"vboxLayout9")
        self.lblGazeDirTitle = QLabel(self.gazeDirFrame)
        self.lblGazeDirTitle.setObjectName(u"lblGazeDirTitle")
        self.lblGazeDirTitle.setMinimumSize(QSize(0, 15))

        self.vboxLayout9.addWidget(self.lblGazeDirTitle)

        self.lblGazeDirVal = QLabel(self.gazeDirFrame)
        self.lblGazeDirVal.setObjectName(u"lblGazeDirVal")

        self.vboxLayout9.addWidget(self.lblGazeDirVal)


        self.sidebarLayout.addWidget(self.gazeDirFrame)

        self.gazeZoneFrame = QFrame(self.sidebarFrame)
        self.gazeZoneFrame.setObjectName(u"gazeZoneFrame")
        self.vboxLayout10 = QVBoxLayout(self.gazeZoneFrame)
        self.vboxLayout10.setObjectName(u"vboxLayout10")
        self.lblGazeZoneTitle = QLabel(self.gazeZoneFrame)
        self.lblGazeZoneTitle.setObjectName(u"lblGazeZoneTitle")

        self.vboxLayout10.addWidget(self.lblGazeZoneTitle)

        self.lblGazeZoneVal = QLabel(self.gazeZoneFrame)
        self.lblGazeZoneVal.setObjectName(u"lblGazeZoneVal")

        self.vboxLayout10.addWidget(self.lblGazeZoneVal)


        self.sidebarLayout.addWidget(self.gazeZoneFrame)

        self.headZoneFrame = QFrame(self.sidebarFrame)
        self.headZoneFrame.setObjectName(u"headZoneFrame")
        self.vboxLayout11 = QVBoxLayout(self.headZoneFrame)
        self.vboxLayout11.setObjectName(u"vboxLayout11")
        self.lblHeadZoneTitle = QLabel(self.headZoneFrame)
        self.lblHeadZoneTitle.setObjectName(u"lblHeadZoneTitle")

        self.vboxLayout11.addWidget(self.lblHeadZoneTitle)

        self.lblHeadZoneVal = QLabel(self.headZoneFrame)
        self.lblHeadZoneVal.setObjectName(u"lblHeadZoneVal")

        self.vboxLayout11.addWidget(self.lblHeadZoneVal)


        self.sidebarLayout.addWidget(self.headZoneFrame)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.sidebarLayout.addItem(self.verticalSpacer)

        self.mainSplitter.addWidget(self.sidebarFrame)
        self.videoContainer = QFrame(self.mainSplitter)
        self.videoContainer.setObjectName(u"videoContainer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.videoContainer.sizePolicy().hasHeightForWidth())
        self.videoContainer.setSizePolicy(sizePolicy1)
        self.videoLayout = QVBoxLayout(self.videoContainer)
        self.videoLayout.setSpacing(0)
        self.videoLayout.setObjectName(u"videoLayout")
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.notifyBar = QLabel(self.videoContainer)
        self.notifyBar.setObjectName(u"notifyBar")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.notifyBar.sizePolicy().hasHeightForWidth())
        self.notifyBar.setSizePolicy(sizePolicy2)
        self.notifyBar.setMinimumSize(QSize(0, 40))
        self.notifyBar.setMaximumSize(QSize(16777215, 40))
        self.notifyBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoLayout.addWidget(self.notifyBar)

        self.videoLabel = QLabel(self.videoContainer)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.videoLabel.sizePolicy().hasHeightForWidth())
        self.videoLabel.setSizePolicy(sizePolicy1)
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoLayout.addWidget(self.videoLabel)

        self.captionBar = QLabel(self.videoContainer)
        self.captionBar.setObjectName(u"captionBar")
        sizePolicy2.setHeightForWidth(self.captionBar.sizePolicy().hasHeightForWidth())
        self.captionBar.setSizePolicy(sizePolicy2)
        self.captionBar.setMinimumSize(QSize(0, 32))
        self.captionBar.setMaximumSize(QSize(16777215, 32))
        self.captionBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoLayout.addWidget(self.captionBar)

        self.mainSplitter.addWidget(self.videoContainer)

        self.centralLayout.addWidget(self.mainSplitter)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Driver Monitoring System", None))
        self.driverIcon.setText("")
        self.driverName.setText(QCoreApplication.translate("MainWindow", u"Chun-Ting", None))
        self.lblDistractionVal.setText(QCoreApplication.translate("MainWindow", u"5%", None))
        self.lblDistractionTitle.setText(QCoreApplication.translate("MainWindow", u"DISTRACTION LEVEL", None))
        self.lblDrowsyVal.setText(QCoreApplication.translate("MainWindow", u"0%", None))
        self.lblDrowsyTitle.setText(QCoreApplication.translate("MainWindow", u"DROWSY LEVEL", None))
        self.lblActionTitle.setText(QCoreApplication.translate("MainWindow", u"REGULAR ACTION", None))
        self.lblActionIcons.setText(QCoreApplication.translate("MainWindow", u"Phone / Drink / Smoke", None))
        self.lblExpTitle.setText(QCoreApplication.translate("MainWindow", u"EYE BLINK", None))
        self.lblExpVal.setText(QCoreApplication.translate("MainWindow", u"NEUTRAL", None))
        self.lblEyeOpTitle.setText(QCoreApplication.translate("MainWindow", u"EYE LOC (mm)", None))
        self.lblEyeOpL.setText(QCoreApplication.translate("MainWindow", u"99%", None))
        self.lblEyeOpR.setText(QCoreApplication.translate("MainWindow", u"99%", None))
        self.lblBlinkTitle.setText(QCoreApplication.translate("MainWindow", u"EYE BLINK", None))
        self.lblBlinkVal.setText(QCoreApplication.translate("MainWindow", u"0.2/s        0.8/s", None))
        self.lblHeadLocTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD LOC (mm)", None))
        self.lblHeadLocVal.setText(QCoreApplication.translate("MainWindow", u"213   -79   546", None))
        self.lblEyeLocTitle.setText(QCoreApplication.translate("MainWindow", u"EYE LOC (mm)", None))
        self.lblEyeLocVal.setText(QCoreApplication.translate("MainWindow", u"L: 233  -77  552\\nR: 192  -81  530", None))
        self.lblHeadDirTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD DIR (PYR)", None))
        self.lblHeadDirVal.setText(QCoreApplication.translate("MainWindow", u"+6\u00b0   +35\u00b0   +3\u00b0", None))
        self.lblGazeDirTitle.setText(QCoreApplication.translate("MainWindow", u"GAZE DIR (PY)", None))
        self.lblGazeDirVal.setText(QCoreApplication.translate("MainWindow", u"+6\u00b0        +39\u00b0", None))
        self.lblGazeZoneTitle.setText(QCoreApplication.translate("MainWindow", u"GAZE ZONE", None))
        self.lblGazeZoneVal.setText(QCoreApplication.translate("MainWindow", u"FRONT_WINDSHIELD", None))
        self.lblHeadZoneTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD ZONE", None))
        self.lblHeadZoneVal.setText(QCoreApplication.translate("MainWindow", u"FRONT_WINDSHIELD", None))
        self.notifyBar.setText("")
        self.videoLabel.setStyleSheet(QCoreApplication.translate("MainWindow", u"background-color: #0d1117;", None))
        self.videoLabel.setText(QCoreApplication.translate("MainWindow", u"Camera Feed", None))
        self.captionBar.setText(QCoreApplication.translate("MainWindow", u"Driver Monitoring System", None))
    # retranslateUi

