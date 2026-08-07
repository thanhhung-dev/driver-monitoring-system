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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSpacerItem, QSplitter, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1280, 1212)
        MainWindow.setStyleSheet(u"QMainWindow {\n"
"    background-color: #1a202c;\n"
"}\n"
"\n"
"QFrame#sidebarFrame {\n"
"    background-color: #293043;\n"
"    border-right: 2px solid #2d3748;\n"
"}\n"
"\n"
"QFrame#profileFrame\n"
"{\n"
"    background-color: #38516f;\n"
"    border: 1px solid #ffffff;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QFrame#distractionFrame,\n"
"QFrame#drowsyFrame {\n"
"    background-color: #38516f;\n"
"    border: 1px solid #ffffff;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QFrame#actionFrame,\n"
"QFrame#expressionFrame,\n"
"QFrame#eyeOpFrame,\n"
"QFrame#blinkFrame,\n"
"QFrame#headLocFrame,\n"
"QFrame#eyeLocFrame,\n"
"QFrame#headDirFrame,\n"
"QFrame#gazeDirFrame,\n"
"QFrame#gazeZoneFrame,\n"
"QFrame#headZoneFrame {\n"
"    border: 0;\n"
"    border-bottom: 1px solid #38516f;\n"
"    padding: 2px;\n"
"}\n"
"\n"
"QLabel {\n"
"    color: #a0aec0;\n"
"    font-family: \"Segoe UI\", Arial, sans-serif;\n"
"    font-size: 11px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QLabel#driverName {\n"
"    color: #ffffff;\n"
""
                        "    border: 0;\n"
"    font-size: 14px;\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QProgressBar#lblDistractionVal,\n"
"QProgressBar#lblDrowsyVal {\n"
"    background-color: #253247;\n"
"    border: 1px solid #ffffff;\n"
"    border-radius: 10px;\n"
"    color: #ffffff;\n"
"    font-size: 12px;\n"
"    font-weight: bold;\n"
"    text-align: center;\n"
"}\n"
"\n"
"QProgressBar#lblDistractionVal::chunk,\n"
"QProgressBar#lblDrowsyVal::chunk {\n"
"    border-radius: 9px;\n"
"    background: qlineargradient(\n"
"        x1: 0, y1: 0, x2: 1, y2: 0,\n"
"        stop: 0 #22c55e,\n"
"        stop: 0.5 #facc15,\n"
"        stop: 1 #ef4444\n"
"    );\n"
"}\n"
"\n"
"QLabel#lblDistractionTitle,\n"
"QLabel#lblDrowsyTitle {\n"
"    color: #ffffff;\n"
"    font-size: 11px;\n"
"}\n"
"\n"
"QLabel#lblActionTitle,\n"
"QLabel#lblExpTitle,\n"
"QLabel#lblEyeOpTitle,\n"
"QLabel#lblBlinkTitle,\n"
"QLabel#lblHeadLocTitle,\n"
"QLabel#lblEyeLocTitle,\n"
"QLabel#lblHeadDirTitle,\n"
"QLabel#lblGazeDirTitle,\n"
"QLabel#lblGazeZoneTitle,\n"
"QL"
                        "abel#lblHeadZoneTitle {\n"
"    color: #83DDDC;\n"
"    font-size: 13px;\n"
"}\n"
"\n"
"QLabel#lblExpVal,\n"
"QLabel#lblEyeOpL,\n"
"QLabel#lblEyeOpR,\n"
"QLabel#lblBlinkVal,\n"
"QLabel#lblBlinkRVal,\n"
"QLabel#lblHeadLocVal,\n"
"QLabel#lblHeadLocYVal,\n"
"QLabel#lblHeadLocZVal,\n"
"QLabel#lblEyeLocLPrefix,\n"
"QLabel#lblEyeLocRPrefix,\n"
"QLabel#lblEyeLocVal,\n"
"QLabel#lblEyeLocLYVal,\n"
"QLabel#lblEyeLocLZVal,\n"
"QLabel#lblEyeLocRXVal,\n"
"QLabel#lblEyeLocRYVal,\n"
"QLabel#lblEyeLocRZVal,\n"
"QLabel#lblHeadDirVal,\n"
"QLabel#lblHeadDirYawVal,\n"
"QLabel#lblHeadDirRollVal,\n"
"QLabel#lblGazeDirVal,\n"
"QLabel#lblGazeDirYawVal,\n"
"QLabel#lblGazeZoneVal,\n"
"QLabel#lblHeadZoneVal {\n"
"    color: #EDA200;\n"
"    border-bottom: 0;\n"
"    font-size: 12px;\n"
"    padding: 2px 0;\n"
"    qproperty-alignment: AlignCenter;\n"
"}\n"
"\n"
"QLabel#notifyBar {\n"
"    background-color: #34394c;\n"
"    color: #fc8181;\n"
"    font-size: 15px;\n"
"}\n"
"\n"
"QLabel#videoLabel {\n"
"    background-color: #0d1117;\n"
""
                        "}\n"
"\n"
"QLabel#captionBar {\n"
"    background-color: #34394a;\n"
"    color: #a0aec0;\n"
"    border-top: 1px solid #2d3748;\n"
"    font-size: 12px;\n"
"    font-weight: normal;\n"
"}\n"
"\n"
"QPushButton {\n"
"    color: #ffffff;\n"
"    border: none;\n"
"    border-radius: 4px;\n"
"    padding: 8px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QLabel#lblRegValue,\n"
"QLabel#lblExpVal {\n"
"    color: #B04A4A;\n"
"    font-size: 12px;\n"
"    min-height: 16px;\n"
"}\n"
"\n"
"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setSpacing(0)
        self.centralLayout.setObjectName(u"centralLayout")
        self.centralLayout.setContentsMargins(0, 0, 0, 0)
        self.mainSplitter = QSplitter(self.centralwidget)
        self.mainSplitter.setObjectName(u"mainSplitter")
        self.mainSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.mainSplitter.setHandleWidth(0)
        self.mainSplitter.setChildrenCollapsible(False)
        self.sidebarFrame = QFrame(self.mainSplitter)
        self.sidebarFrame.setObjectName(u"sidebarFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sidebarFrame.sizePolicy().hasHeightForWidth())
        self.sidebarFrame.setSizePolicy(sizePolicy)
        self.sidebarFrame.setMinimumSize(QSize(180, 0))
        self.sidebarFrame.setMaximumSize(QSize(180, 16777215))
        self.sidebarLayout = QVBoxLayout(self.sidebarFrame)
        self.sidebarLayout.setSpacing(8)
        self.sidebarLayout.setObjectName(u"sidebarLayout")
        self.sidebarLayout.setContentsMargins(0, 10, 0, 10)
        self.userFrame = QFrame(self.sidebarFrame)
        self.userFrame.setObjectName(u"userFrame")
        self.userFrame.setMinimumSize(QSize(0, 25))
        self.userLayout = QHBoxLayout(self.userFrame)
        self.userLayout.setSpacing(10)
        self.userLayout.setObjectName(u"userLayout")
        self.userLayout.setContentsMargins(12, 6, 12, 6)

        self.sidebarLayout.addWidget(self.userFrame)

        self.profileFrame = QFrame(self.sidebarFrame)
        self.profileFrame.setObjectName(u"profileFrame")
        self.profileFrame.setMinimumSize(QSize(0, 70))
        self.profileLayout = QVBoxLayout(self.profileFrame)
        self.profileLayout.setSpacing(4)
        self.profileLayout.setObjectName(u"profileLayout")
        self.profileLayout.setContentsMargins(8, 6, 8, 6)
        self.statusIconRow = QHBoxLayout()
        self.statusIconRow.setSpacing(10)
        self.statusIconRow.setObjectName(u"statusIconRow")
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.statusIconRow.addItem(self.spacer)

        self.iconGlasses = QLabel(self.profileFrame)
        self.iconGlasses.setObjectName(u"iconGlasses")
        self.iconGlasses.setMinimumSize(QSize(28, 28))
        self.iconGlasses.setMaximumSize(QSize(28, 28))
        self.iconGlasses.setScaledContents(True)
        self.iconGlasses.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.statusIconRow.addWidget(self.iconGlasses)

        self.iconDevice = QLabel(self.profileFrame)
        self.iconDevice.setObjectName(u"iconDevice")
        self.iconDevice.setMinimumSize(QSize(28, 28))
        self.iconDevice.setMaximumSize(QSize(28, 28))
        self.iconDevice.setScaledContents(True)
        self.iconDevice.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.statusIconRow.addWidget(self.iconDevice)

        self.iconPerson = QLabel(self.profileFrame)
        self.iconPerson.setObjectName(u"iconPerson")
        self.iconPerson.setMinimumSize(QSize(28, 28))
        self.iconPerson.setMaximumSize(QSize(28, 28))
        self.iconPerson.setScaledContents(True)
        self.iconPerson.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.statusIconRow.addWidget(self.iconPerson)


        self.profileLayout.addLayout(self.statusIconRow)

        self.driverName = QLabel(self.profileFrame)
        self.driverName.setObjectName(u"driverName")
        self.driverName.setMinimumSize(QSize(0, 20))
        self.driverName.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.profileLayout.addWidget(self.driverName)


        self.sidebarLayout.addWidget(self.profileFrame)

        self.distractionFrame = QFrame(self.sidebarFrame)
        self.distractionFrame.setObjectName(u"distractionFrame")
        self.distractionFrame.setMaximumSize(QSize(16777215, 80))
        self.distractionStack = QGridLayout(self.distractionFrame)
        self.distractionStack.setObjectName(u"distractionStack")
        self.distractionStack.setProperty(u"currentIndex", 1)
        self.distractionStack.setContentsMargins(0, 0, 0, 0)
        self.lblDistractionVal = QProgressBar(self.distractionFrame)
        self.lblDistractionVal.setObjectName(u"lblDistractionVal")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lblDistractionVal.sizePolicy().hasHeightForWidth())
        self.lblDistractionVal.setSizePolicy(sizePolicy1)
        self.lblDistractionVal.setMaximum(80)
        self.lblDistractionVal.setValue(5)
        self.lblDistractionVal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblDistractionVal.setTextVisible(True)

        self.distractionStack.addWidget(self.lblDistractionVal, 0, 0, 1, 1)

        self.distractionTextLayout = QVBoxLayout()
        self.distractionTextLayout.setObjectName(u"distractionTextLayout")
        self.distractionTextLayout.setContentsMargins(0, 0, 0, 5)
        self.distractionTitleSpacer = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.distractionTextLayout.addItem(self.distractionTitleSpacer)

        self.lblDistractionTitle = QLabel(self.distractionFrame)
        self.lblDistractionTitle.setObjectName(u"lblDistractionTitle")
        self.lblDistractionTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.distractionTextLayout.addWidget(self.lblDistractionTitle)


        self.distractionStack.addLayout(self.distractionTextLayout, 0, 0, 1, 1)


        self.sidebarLayout.addWidget(self.distractionFrame)

        self.drowsyFrame = QFrame(self.sidebarFrame)
        self.drowsyFrame.setObjectName(u"drowsyFrame")
        self.drowsyFrame.setMaximumSize(QSize(16777215, 80))
        self.drowsyStack = QGridLayout(self.drowsyFrame)
        self.drowsyStack.setObjectName(u"drowsyStack")
        self.drowsyStack.setProperty(u"currentIndex", 1)
        self.drowsyStack.setContentsMargins(0, 0, 0, 0)
        self.lblDrowsyVal = QProgressBar(self.drowsyFrame)
        self.lblDrowsyVal.setObjectName(u"lblDrowsyVal")
        sizePolicy1.setHeightForWidth(self.lblDrowsyVal.sizePolicy().hasHeightForWidth())
        self.lblDrowsyVal.setSizePolicy(sizePolicy1)
        self.lblDrowsyVal.setMaximum(100)
        self.lblDrowsyVal.setValue(0)
        self.lblDrowsyVal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblDrowsyVal.setTextVisible(True)

        self.drowsyStack.addWidget(self.lblDrowsyVal, 0, 0, 1, 1)

        self.drowsyTextLayout = QVBoxLayout()
        self.drowsyTextLayout.setObjectName(u"drowsyTextLayout")
        self.drowsyTextLayout.setContentsMargins(0, 0, 0, 5)
        self.drowsyTitleSpacer = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.drowsyTextLayout.addItem(self.drowsyTitleSpacer)

        self.lblDrowsyTitle = QLabel(self.drowsyFrame)
        self.lblDrowsyTitle.setObjectName(u"lblDrowsyTitle")
        self.lblDrowsyTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.drowsyTextLayout.addWidget(self.lblDrowsyTitle)


        self.drowsyStack.addLayout(self.drowsyTextLayout, 0, 0, 1, 1)


        self.sidebarLayout.addWidget(self.drowsyFrame)

        self.actionFrame = QFrame(self.sidebarFrame)
        self.actionFrame.setObjectName(u"actionFrame")
        self.actionLayout = QVBoxLayout(self.actionFrame)
        self.actionLayout.setObjectName(u"actionLayout")
        self.actionLayout.setContentsMargins(0, -1, 0, -1)
        self.actionTitleRow = QHBoxLayout()
        self.actionTitleRow.setObjectName(u"actionTitleRow")
        self.iconActionTitle = QLabel(self.actionFrame)
        self.iconActionTitle.setObjectName(u"iconActionTitle")

        self.actionTitleRow.addWidget(self.iconActionTitle)

        self.lblActionTitle = QLabel(self.actionFrame)
        self.lblActionTitle.setObjectName(u"lblActionTitle")

        self.actionTitleRow.addWidget(self.lblActionTitle)

        self.spacerItem = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.actionTitleRow.addItem(self.spacerItem)


        self.actionLayout.addLayout(self.actionTitleRow)

        self.lblRegValue = QLabel(self.actionFrame)
        self.lblRegValue.setObjectName(u"lblRegValue")
        self.lblRegValue.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.actionLayout.addWidget(self.lblRegValue)

        self.actionIconRow = QHBoxLayout()
        self.actionIconRow.setSpacing(8)
        self.actionIconRow.setObjectName(u"actionIconRow")
        self.iconActionPhone = QLabel(self.actionFrame)
        self.iconActionPhone.setObjectName(u"iconActionPhone")
        self.iconActionPhone.setMinimumSize(QSize(24, 24))
        self.iconActionPhone.setMaximumSize(QSize(24, 24))
        self.iconActionPhone.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.actionIconRow.addWidget(self.iconActionPhone)

        self.iconActionDrink = QLabel(self.actionFrame)
        self.iconActionDrink.setObjectName(u"iconActionDrink")
        self.iconActionDrink.setMinimumSize(QSize(24, 24))
        self.iconActionDrink.setMaximumSize(QSize(24, 24))
        self.iconActionDrink.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.actionIconRow.addWidget(self.iconActionDrink)

        self.iconActionSmoke = QLabel(self.actionFrame)
        self.iconActionSmoke.setObjectName(u"iconActionSmoke")
        self.iconActionSmoke.setMinimumSize(QSize(24, 24))
        self.iconActionSmoke.setMaximumSize(QSize(24, 24))
        self.iconActionSmoke.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.actionIconRow.addWidget(self.iconActionSmoke)

        self.iconActionYawn = QLabel(self.actionFrame)
        self.iconActionYawn.setObjectName(u"iconActionYawn")
        self.iconActionYawn.setMinimumSize(QSize(24, 24))
        self.iconActionYawn.setMaximumSize(QSize(24, 24))
        self.iconActionYawn.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.actionIconRow.addWidget(self.iconActionYawn)


        self.actionLayout.addLayout(self.actionIconRow)


        self.sidebarLayout.addWidget(self.actionFrame)

        self.expressionFrame = QFrame(self.sidebarFrame)
        self.expressionFrame.setObjectName(u"expressionFrame")
        self.vboxLayout = QVBoxLayout(self.expressionFrame)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.vboxLayout.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout = QHBoxLayout()
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.iconExpression = QLabel(self.expressionFrame)
        self.iconExpression.setObjectName(u"iconExpression")

        self.hboxLayout.addWidget(self.iconExpression)

        self.lblExpTitle = QLabel(self.expressionFrame)
        self.lblExpTitle.setObjectName(u"lblExpTitle")

        self.hboxLayout.addWidget(self.lblExpTitle)

        self.spacerItem1 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout.addItem(self.spacerItem1)


        self.vboxLayout.addLayout(self.hboxLayout)

        self.lblExpVal = QLabel(self.expressionFrame)
        self.lblExpVal.setObjectName(u"lblExpVal")
        self.lblExpVal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vboxLayout.addWidget(self.lblExpVal)


        self.sidebarLayout.addWidget(self.expressionFrame)

        self.eyeOpFrame = QFrame(self.sidebarFrame)
        self.eyeOpFrame.setObjectName(u"eyeOpFrame")
        self.vboxLayout1 = QVBoxLayout(self.eyeOpFrame)
        self.vboxLayout1.setObjectName(u"vboxLayout1")
        self.vboxLayout1.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout1 = QHBoxLayout()
        self.hboxLayout1.setObjectName(u"hboxLayout1")
        self.iconEyeOpenness = QLabel(self.eyeOpFrame)
        self.iconEyeOpenness.setObjectName(u"iconEyeOpenness")

        self.hboxLayout1.addWidget(self.iconEyeOpenness)

        self.lblEyeOpTitle = QLabel(self.eyeOpFrame)
        self.lblEyeOpTitle.setObjectName(u"lblEyeOpTitle")

        self.hboxLayout1.addWidget(self.lblEyeOpTitle)

        self.spacerItem2 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout1.addItem(self.spacerItem2)


        self.vboxLayout1.addLayout(self.hboxLayout1)

        self.eyeOpValueRow = QHBoxLayout()
        self.eyeOpValueRow.setSpacing(12)
        self.eyeOpValueRow.setObjectName(u"eyeOpValueRow")
        self.lblEyeOpL = QLabel(self.eyeOpFrame)
        self.lblEyeOpL.setObjectName(u"lblEyeOpL")
        self.lblEyeOpL.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.eyeOpValueRow.addWidget(self.lblEyeOpL)

        self.lblEyeOpR = QLabel(self.eyeOpFrame)
        self.lblEyeOpR.setObjectName(u"lblEyeOpR")
        self.lblEyeOpR.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.eyeOpValueRow.addWidget(self.lblEyeOpR)

        self.eyeOpValueRow.setStretch(0, 1)
        self.eyeOpValueRow.setStretch(1, 1)

        self.vboxLayout1.addLayout(self.eyeOpValueRow)


        self.sidebarLayout.addWidget(self.eyeOpFrame)

        self.blinkFrame = QFrame(self.sidebarFrame)
        self.blinkFrame.setObjectName(u"blinkFrame")
        self.vboxLayout2 = QVBoxLayout(self.blinkFrame)
        self.vboxLayout2.setObjectName(u"vboxLayout2")
        self.vboxLayout2.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout2 = QHBoxLayout()
        self.hboxLayout2.setObjectName(u"hboxLayout2")
        self.iconEyeBlink = QLabel(self.blinkFrame)
        self.iconEyeBlink.setObjectName(u"iconEyeBlink")

        self.hboxLayout2.addWidget(self.iconEyeBlink)

        self.lblBlinkTitle = QLabel(self.blinkFrame)
        self.lblBlinkTitle.setObjectName(u"lblBlinkTitle")

        self.hboxLayout2.addWidget(self.lblBlinkTitle)

        self.spacerItem3 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout2.addItem(self.spacerItem3)


        self.vboxLayout2.addLayout(self.hboxLayout2)

        self.blinkValueRow = QHBoxLayout()
        self.blinkValueRow.setSpacing(12)
        self.blinkValueRow.setObjectName(u"blinkValueRow")
        self.lblBlinkVal = QLabel(self.blinkFrame)
        self.lblBlinkVal.setObjectName(u"lblBlinkVal")

        self.blinkValueRow.addWidget(self.lblBlinkVal)

        self.lblBlinkRVal = QLabel(self.blinkFrame)
        self.lblBlinkRVal.setObjectName(u"lblBlinkRVal")

        self.blinkValueRow.addWidget(self.lblBlinkRVal)

        self.blinkValueRow.setStretch(0, 1)
        self.blinkValueRow.setStretch(1, 1)

        self.vboxLayout2.addLayout(self.blinkValueRow)


        self.sidebarLayout.addWidget(self.blinkFrame)

        self.headLocFrame = QFrame(self.sidebarFrame)
        self.headLocFrame.setObjectName(u"headLocFrame")
        self.vboxLayout3 = QVBoxLayout(self.headLocFrame)
        self.vboxLayout3.setObjectName(u"vboxLayout3")
        self.vboxLayout3.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout3 = QHBoxLayout()
        self.hboxLayout3.setObjectName(u"hboxLayout3")
        self.iconHeadLoc = QLabel(self.headLocFrame)
        self.iconHeadLoc.setObjectName(u"iconHeadLoc")

        self.hboxLayout3.addWidget(self.iconHeadLoc)

        self.lblHeadLocTitle = QLabel(self.headLocFrame)
        self.lblHeadLocTitle.setObjectName(u"lblHeadLocTitle")

        self.hboxLayout3.addWidget(self.lblHeadLocTitle)

        self.spacerItem4 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout3.addItem(self.spacerItem4)


        self.vboxLayout3.addLayout(self.hboxLayout3)

        self.headLocValueRow = QHBoxLayout()
        self.headLocValueRow.setSpacing(8)
        self.headLocValueRow.setObjectName(u"headLocValueRow")
        self.lblHeadLocVal = QLabel(self.headLocFrame)
        self.lblHeadLocVal.setObjectName(u"lblHeadLocVal")

        self.headLocValueRow.addWidget(self.lblHeadLocVal)

        self.lblHeadLocYVal = QLabel(self.headLocFrame)
        self.lblHeadLocYVal.setObjectName(u"lblHeadLocYVal")

        self.headLocValueRow.addWidget(self.lblHeadLocYVal)

        self.lblHeadLocZVal = QLabel(self.headLocFrame)
        self.lblHeadLocZVal.setObjectName(u"lblHeadLocZVal")

        self.headLocValueRow.addWidget(self.lblHeadLocZVal)

        self.headLocValueRow.setStretch(0, 1)
        self.headLocValueRow.setStretch(1, 1)
        self.headLocValueRow.setStretch(2, 1)

        self.vboxLayout3.addLayout(self.headLocValueRow)


        self.sidebarLayout.addWidget(self.headLocFrame)

        self.eyeLocFrame = QFrame(self.sidebarFrame)
        self.eyeLocFrame.setObjectName(u"eyeLocFrame")
        self.vboxLayout4 = QVBoxLayout(self.eyeLocFrame)
        self.vboxLayout4.setObjectName(u"vboxLayout4")
        self.vboxLayout4.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout4 = QHBoxLayout()
        self.hboxLayout4.setObjectName(u"hboxLayout4")
        self.iconEyeLoc = QLabel(self.eyeLocFrame)
        self.iconEyeLoc.setObjectName(u"iconEyeLoc")

        self.hboxLayout4.addWidget(self.iconEyeLoc)

        self.lblEyeLocTitle = QLabel(self.eyeLocFrame)
        self.lblEyeLocTitle.setObjectName(u"lblEyeLocTitle")

        self.hboxLayout4.addWidget(self.lblEyeLocTitle)

        self.spacerItem5 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout4.addItem(self.spacerItem5)


        self.vboxLayout4.addLayout(self.hboxLayout4)

        self.eyeLocValueGrid = QGridLayout()
        self.eyeLocValueGrid.setObjectName(u"eyeLocValueGrid")
        self.eyeLocValueGrid.setHorizontalSpacing(4)
        self.eyeLocValueGrid.setVerticalSpacing(2)
        self.lblEyeLocLPrefix = QLabel(self.eyeLocFrame)
        self.lblEyeLocLPrefix.setObjectName(u"lblEyeLocLPrefix")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocLPrefix, 0, 0, 1, 1)

        self.lblEyeLocVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocVal.setObjectName(u"lblEyeLocVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocVal, 0, 1, 1, 1)

        self.lblEyeLocLYVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocLYVal.setObjectName(u"lblEyeLocLYVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocLYVal, 0, 2, 1, 1)

        self.lblEyeLocLZVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocLZVal.setObjectName(u"lblEyeLocLZVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocLZVal, 0, 3, 1, 1)

        self.lblEyeLocRPrefix = QLabel(self.eyeLocFrame)
        self.lblEyeLocRPrefix.setObjectName(u"lblEyeLocRPrefix")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocRPrefix, 1, 0, 1, 1)

        self.lblEyeLocRXVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocRXVal.setObjectName(u"lblEyeLocRXVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocRXVal, 1, 1, 1, 1)

        self.lblEyeLocRYVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocRYVal.setObjectName(u"lblEyeLocRYVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocRYVal, 1, 2, 1, 1)

        self.lblEyeLocRZVal = QLabel(self.eyeLocFrame)
        self.lblEyeLocRZVal.setObjectName(u"lblEyeLocRZVal")

        self.eyeLocValueGrid.addWidget(self.lblEyeLocRZVal, 1, 3, 1, 1)

        self.eyeLocValueGrid.setColumnStretch(1, 1)
        self.eyeLocValueGrid.setColumnStretch(2, 1)
        self.eyeLocValueGrid.setColumnStretch(3, 1)

        self.vboxLayout4.addLayout(self.eyeLocValueGrid)


        self.sidebarLayout.addWidget(self.eyeLocFrame)

        self.headDirFrame = QFrame(self.sidebarFrame)
        self.headDirFrame.setObjectName(u"headDirFrame")
        self.vboxLayout5 = QVBoxLayout(self.headDirFrame)
        self.vboxLayout5.setObjectName(u"vboxLayout5")
        self.vboxLayout5.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout5 = QHBoxLayout()
        self.hboxLayout5.setObjectName(u"hboxLayout5")
        self.iconHeadDir = QLabel(self.headDirFrame)
        self.iconHeadDir.setObjectName(u"iconHeadDir")

        self.hboxLayout5.addWidget(self.iconHeadDir)

        self.lblHeadDirTitle = QLabel(self.headDirFrame)
        self.lblHeadDirTitle.setObjectName(u"lblHeadDirTitle")

        self.hboxLayout5.addWidget(self.lblHeadDirTitle)

        self.spacerItem6 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout5.addItem(self.spacerItem6)


        self.vboxLayout5.addLayout(self.hboxLayout5)

        self.headDirValueRow = QHBoxLayout()
        self.headDirValueRow.setSpacing(8)
        self.headDirValueRow.setObjectName(u"headDirValueRow")
        self.lblHeadDirVal = QLabel(self.headDirFrame)
        self.lblHeadDirVal.setObjectName(u"lblHeadDirVal")

        self.headDirValueRow.addWidget(self.lblHeadDirVal)

        self.lblHeadDirYawVal = QLabel(self.headDirFrame)
        self.lblHeadDirYawVal.setObjectName(u"lblHeadDirYawVal")

        self.headDirValueRow.addWidget(self.lblHeadDirYawVal)

        self.lblHeadDirRollVal = QLabel(self.headDirFrame)
        self.lblHeadDirRollVal.setObjectName(u"lblHeadDirRollVal")

        self.headDirValueRow.addWidget(self.lblHeadDirRollVal)

        self.headDirValueRow.setStretch(0, 1)
        self.headDirValueRow.setStretch(1, 1)
        self.headDirValueRow.setStretch(2, 1)

        self.vboxLayout5.addLayout(self.headDirValueRow)


        self.sidebarLayout.addWidget(self.headDirFrame)

        self.gazeDirFrame = QFrame(self.sidebarFrame)
        self.gazeDirFrame.setObjectName(u"gazeDirFrame")
        self.vboxLayout6 = QVBoxLayout(self.gazeDirFrame)
        self.vboxLayout6.setObjectName(u"vboxLayout6")
        self.vboxLayout6.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout6 = QHBoxLayout()
        self.hboxLayout6.setObjectName(u"hboxLayout6")
        self.iconGazeDir = QLabel(self.gazeDirFrame)
        self.iconGazeDir.setObjectName(u"iconGazeDir")

        self.hboxLayout6.addWidget(self.iconGazeDir)

        self.lblGazeDirTitle = QLabel(self.gazeDirFrame)
        self.lblGazeDirTitle.setObjectName(u"lblGazeDirTitle")
        self.lblGazeDirTitle.setMinimumSize(QSize(0, 15))

        self.hboxLayout6.addWidget(self.lblGazeDirTitle)

        self.spacerItem7 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout6.addItem(self.spacerItem7)


        self.vboxLayout6.addLayout(self.hboxLayout6)

        self.gazeDirValueRow = QHBoxLayout()
        self.gazeDirValueRow.setSpacing(12)
        self.gazeDirValueRow.setObjectName(u"gazeDirValueRow")
        self.lblGazeDirVal = QLabel(self.gazeDirFrame)
        self.lblGazeDirVal.setObjectName(u"lblGazeDirVal")

        self.gazeDirValueRow.addWidget(self.lblGazeDirVal)

        self.lblGazeDirYawVal = QLabel(self.gazeDirFrame)
        self.lblGazeDirYawVal.setObjectName(u"lblGazeDirYawVal")

        self.gazeDirValueRow.addWidget(self.lblGazeDirYawVal)

        self.gazeDirValueRow.setStretch(0, 1)
        self.gazeDirValueRow.setStretch(1, 1)

        self.vboxLayout6.addLayout(self.gazeDirValueRow)


        self.sidebarLayout.addWidget(self.gazeDirFrame)

        self.gazeZoneFrame = QFrame(self.sidebarFrame)
        self.gazeZoneFrame.setObjectName(u"gazeZoneFrame")
        self.vboxLayout7 = QVBoxLayout(self.gazeZoneFrame)
        self.vboxLayout7.setObjectName(u"vboxLayout7")
        self.vboxLayout7.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout7 = QHBoxLayout()
        self.hboxLayout7.setObjectName(u"hboxLayout7")
        self.iconGazeZone = QLabel(self.gazeZoneFrame)
        self.iconGazeZone.setObjectName(u"iconGazeZone")

        self.hboxLayout7.addWidget(self.iconGazeZone)

        self.lblGazeZoneTitle = QLabel(self.gazeZoneFrame)
        self.lblGazeZoneTitle.setObjectName(u"lblGazeZoneTitle")

        self.hboxLayout7.addWidget(self.lblGazeZoneTitle)

        self.spacerItem8 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout7.addItem(self.spacerItem8)


        self.vboxLayout7.addLayout(self.hboxLayout7)

        self.lblGazeZoneVal = QLabel(self.gazeZoneFrame)
        self.lblGazeZoneVal.setObjectName(u"lblGazeZoneVal")

        self.vboxLayout7.addWidget(self.lblGazeZoneVal)


        self.sidebarLayout.addWidget(self.gazeZoneFrame)

        self.headZoneFrame = QFrame(self.sidebarFrame)
        self.headZoneFrame.setObjectName(u"headZoneFrame")
        self.vboxLayout8 = QVBoxLayout(self.headZoneFrame)
        self.vboxLayout8.setObjectName(u"vboxLayout8")
        self.vboxLayout8.setContentsMargins(0, -1, 0, -1)
        self.hboxLayout8 = QHBoxLayout()
        self.hboxLayout8.setObjectName(u"hboxLayout8")
        self.iconHeadZone = QLabel(self.headZoneFrame)
        self.iconHeadZone.setObjectName(u"iconHeadZone")

        self.hboxLayout8.addWidget(self.iconHeadZone)

        self.lblHeadZoneTitle = QLabel(self.headZoneFrame)
        self.lblHeadZoneTitle.setObjectName(u"lblHeadZoneTitle")

        self.hboxLayout8.addWidget(self.lblHeadZoneTitle)

        self.spacerItem9 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hboxLayout8.addItem(self.spacerItem9)


        self.vboxLayout8.addLayout(self.hboxLayout8)

        self.lblHeadZoneVal = QLabel(self.headZoneFrame)
        self.lblHeadZoneVal.setObjectName(u"lblHeadZoneVal")

        self.vboxLayout8.addWidget(self.lblHeadZoneVal)


        self.sidebarLayout.addWidget(self.headZoneFrame)

        self.bottomButtonRow = QHBoxLayout()
        self.bottomButtonRow.setSpacing(16)
        self.bottomButtonRow.setObjectName(u"bottomButtonRow")
        self.spacerItem10 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottomButtonRow.addItem(self.spacerItem10)

        self.btnAddDriver = QPushButton(self.sidebarFrame)
        self.btnAddDriver.setObjectName(u"btnAddDriver")
        self.btnAddDriver.setMinimumSize(QSize(40, 40))
        self.btnAddDriver.setIconSize(QSize(24, 24))

        self.bottomButtonRow.addWidget(self.btnAddDriver)

        self.btnLogout = QPushButton(self.sidebarFrame)
        self.btnLogout.setObjectName(u"btnLogout")
        self.btnLogout.setMinimumSize(QSize(40, 40))
        self.btnLogout.setIconSize(QSize(24, 24))

        self.bottomButtonRow.addWidget(self.btnLogout)

        self.spacerItem11 = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottomButtonRow.addItem(self.spacerItem11)


        self.sidebarLayout.addLayout(self.bottomButtonRow)

        self.mainSplitter.addWidget(self.sidebarFrame)
        self.videoContainer = QFrame(self.mainSplitter)
        self.videoContainer.setObjectName(u"videoContainer")
        self.videoContainer.setEnabled(True)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.videoContainer.sizePolicy().hasHeightForWidth())
        self.videoContainer.setSizePolicy(sizePolicy2)
        self.videoContainer.setMinimumSize(QSize(0, 0))
        self.videoLayout = QVBoxLayout(self.videoContainer)
        self.videoLayout.setSpacing(0)
        self.videoLayout.setObjectName(u"videoLayout")
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.notifyBar = QLabel(self.videoContainer)
        self.notifyBar.setObjectName(u"notifyBar")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.notifyBar.sizePolicy().hasHeightForWidth())
        self.notifyBar.setSizePolicy(sizePolicy3)
        self.notifyBar.setMinimumSize(QSize(0, 26))
        self.notifyBar.setMaximumSize(QSize(16777215, 40))
        self.notifyBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoLayout.addWidget(self.notifyBar)

        self.videoLabel = QLabel(self.videoContainer)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.videoLabel.sizePolicy().hasHeightForWidth())
        self.videoLabel.setSizePolicy(sizePolicy2)
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoLayout.addWidget(self.videoLabel)

        self.captionBar = QLabel(self.videoContainer)
        self.captionBar.setObjectName(u"captionBar")
        sizePolicy3.setHeightForWidth(self.captionBar.sizePolicy().hasHeightForWidth())
        self.captionBar.setSizePolicy(sizePolicy3)
        self.captionBar.setMinimumSize(QSize(0, 32))
        self.captionBar.setMaximumSize(QSize(16777215, 32))
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setBold(False)
        self.captionBar.setFont(font)
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
        self.iconGlasses.setText("")
        self.iconDevice.setText("")
        self.iconPerson.setText("")
        self.driverName.setText(QCoreApplication.translate("MainWindow", u"Name Driver", None))
        self.lblDistractionVal.setFormat(QCoreApplication.translate("MainWindow", u"%p%", None))
        self.lblDistractionTitle.setText(QCoreApplication.translate("MainWindow", u"DISTRACTION LEVEL", None))
        self.lblDrowsyVal.setFormat(QCoreApplication.translate("MainWindow", u"%p%", None))
        self.lblDrowsyTitle.setText(QCoreApplication.translate("MainWindow", u"DROWSY LEVEL", None))
        self.iconActionTitle.setText("")
        self.lblActionTitle.setText(QCoreApplication.translate("MainWindow", u"REGULAR ACTION", None))
        self.lblRegValue.setText(QCoreApplication.translate("MainWindow", u"TEXTLABLE", None))
        self.iconActionPhone.setText("")
        self.iconActionDrink.setText("")
        self.iconActionSmoke.setText("")
        self.iconActionYawn.setText("")
        self.iconExpression.setText("")
        self.lblExpTitle.setText(QCoreApplication.translate("MainWindow", u"EXPRESSION", None))
        self.lblExpVal.setText(QCoreApplication.translate("MainWindow", u"NEUTRAL", None))
        self.iconEyeOpenness.setText("")
        self.lblEyeOpTitle.setText(QCoreApplication.translate("MainWindow", u"EYE OPENNESS", None))
        self.lblEyeOpL.setText(QCoreApplication.translate("MainWindow", u"70%", None))
        self.lblEyeOpR.setText(QCoreApplication.translate("MainWindow", u"70%", None))
        self.iconEyeBlink.setText("")
        self.lblBlinkTitle.setText(QCoreApplication.translate("MainWindow", u"EYE BLINK", None))
        self.lblBlinkVal.setText(QCoreApplication.translate("MainWindow", u"0.2/s", None))
        self.lblBlinkRVal.setText(QCoreApplication.translate("MainWindow", u"0.8/s", None))
        self.iconHeadLoc.setText("")
        self.lblHeadLocTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD LOC (mm)", None))
        self.lblHeadLocVal.setText(QCoreApplication.translate("MainWindow", u"213", None))
        self.lblHeadLocYVal.setText(QCoreApplication.translate("MainWindow", u"-79", None))
        self.lblHeadLocZVal.setText(QCoreApplication.translate("MainWindow", u"546", None))
        self.iconEyeLoc.setText("")
        self.lblEyeLocTitle.setText(QCoreApplication.translate("MainWindow", u"EYE LOC (mm)", None))
        self.lblEyeLocLPrefix.setText(QCoreApplication.translate("MainWindow", u"L:", None))
        self.lblEyeLocVal.setText(QCoreApplication.translate("MainWindow", u"233", None))
        self.lblEyeLocLYVal.setText(QCoreApplication.translate("MainWindow", u"-77", None))
        self.lblEyeLocLZVal.setText(QCoreApplication.translate("MainWindow", u"552", None))
        self.lblEyeLocRPrefix.setText(QCoreApplication.translate("MainWindow", u"R:", None))
        self.lblEyeLocRXVal.setText(QCoreApplication.translate("MainWindow", u"192", None))
        self.lblEyeLocRYVal.setText(QCoreApplication.translate("MainWindow", u"-81", None))
        self.lblEyeLocRZVal.setText(QCoreApplication.translate("MainWindow", u"530", None))
        self.iconHeadDir.setText("")
        self.lblHeadDirTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD DIR (PYR)", None))
        self.lblHeadDirVal.setText(QCoreApplication.translate("MainWindow", u"+6\u00b0", None))
        self.lblHeadDirYawVal.setText(QCoreApplication.translate("MainWindow", u"+35\u00b0", None))
        self.lblHeadDirRollVal.setText(QCoreApplication.translate("MainWindow", u"+3\u00b0", None))
        self.iconGazeDir.setText("")
        self.lblGazeDirTitle.setText(QCoreApplication.translate("MainWindow", u"GAZE DIR (PY)", None))
        self.lblGazeDirVal.setText(QCoreApplication.translate("MainWindow", u"+6\u00b0", None))
        self.lblGazeDirYawVal.setText(QCoreApplication.translate("MainWindow", u"+39\u00b0", None))
        self.iconGazeZone.setText("")
        self.lblGazeZoneTitle.setText(QCoreApplication.translate("MainWindow", u"GAZE ZONE", None))
        self.lblGazeZoneVal.setText(QCoreApplication.translate("MainWindow", u"FRONT_WINDSHIELD", None))
        self.iconHeadZone.setText("")
        self.lblHeadZoneTitle.setText(QCoreApplication.translate("MainWindow", u"HEAD ZONE", None))
        self.lblHeadZoneVal.setText(QCoreApplication.translate("MainWindow", u"FRONT_WINDSHIELD", None))
        self.btnAddDriver.setText("")
        self.btnLogout.setText("")
        self.notifyBar.setText("")
        self.videoLabel.setText(QCoreApplication.translate("MainWindow", u"Camera Feed", None))
        self.captionBar.setText(QCoreApplication.translate("MainWindow", u"Driver Monitoring System", None))
    # retranslateUi

