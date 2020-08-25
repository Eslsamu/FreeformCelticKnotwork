from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget

class ImageArea(QWidget):

    image = None

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        """
        Sets our default image with the right size filled in white.
        """
        self.background = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.background.fill(Qt.white)

        # init painter
        self.qp = QPainter()


    """
    Method called when a painting event occurs.
    """
    def paintEvent(self, event):
        self.qp.begin(self)
        self.qp.drawImage(self.rect(), self.background, self.background.rect())

        # draw the image
        if self.parent.display_img_file:
            self.image = QImage(self.parent.display_img_file)
            self.setMinimumHeight(self.image.height())
            self.setMinimumWidth(self.image.width())
            self.qp.drawImage(QPoint(0,0), self.image)

        self.qp.end()