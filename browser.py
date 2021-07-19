import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import *

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://google.com'))
        self.setCentralWidget(self.browser)
        self.showMaximized()
        #navbar
        navbar = QToolBar()
        self.addToolBar(navbar)

        back_btn = QAction('Back', self)
        back_btn.triggered.connect(self.browser.back)
        navbar.addAction(back_btn)

        forward_btn = QAction('Forward', self)
        forward_btn.triggered.connect(self.browser.forward)
        navbar.addAction(forward_btn)

        refresh_btn = QAction('Refresh', self)
        refresh_btn.triggered.connect(self.browser.reload)
        navbar.addAction(refresh_btn)

        home_btn = QAction('Home', self)
        home_btn.triggered.connect(self.go_home)
        navbar.addAction(home_btn)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.go_toUrl)

        navbar.addWidget(self.url_bar)

        self.browser.urlChanged.connect(self.update_searchbox)

    def go_home(self):
        self.browser.setUrl(QUrl('http://google.com'))
    def update_searchbox(self):
        self.url_bar.setText(self.browser.url().toString())
    def go_toUrl(self):
        url = self.url_bar.text()
        if url.__contains__('http') and url.__contains__('.com'):
            self.browser.setUrl(QUrl(url))
        elif url.__contains__('http'):
            url = url +".com"
            self.url_bar.setText(url)
            self.browser.setUrl(QUrl(url))
        elif url.__contains__('.com'):
            url = "http://"+url
            self.url_bar.setText(url)
            self.browser.setUrl(QUrl(url))
        else:
            url = "http://" + url + ".com"
            self.url_bar.setText(url)
            self.browser.setUrl(QUrl(url))



app = QApplication(sys.argv)

QApplication.setApplicationName("CBMIR")
window = MainWindow()

app.exec()