from mercat import compute_curves
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget


class KnotArea(QWidget):

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
        self.black_pen = QPen(Qt.black, self.parent.knot_width / 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.parent.knot_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.parent.knot_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        self.qp.begin(self)
        self.qp.drawImage(self.rect(), self.background, self.background.rect())

        # draw the knot
        if self.parent.display_knot:
            self.draw_knot()

        # adjust widget size based on loaded image
        img = self.parent.imgArea.image
        if img is not None:
            self.setMinimumWidth(img.width() * 1.2)
            self.setMinimumHeight(img.height() * 1.2)

        self.qp.end()

    def draw_knot(self):
        curves = compute_curves(self.parent.graph["vertices"], self.parent.graph["edges"])
        for p0, p1, p2, p3 in curves:
            path = QPainterPath(QPoint(p0[0], p0[1]))
            path.cubicTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            self.qp.setPen(self.green_pen)
            self.qp.drawPath(path)
            if self.parent.display_control_points:
                self.qp.setPen(self.red_pen)
                self.qp.drawPoint(p1[0], p1[1])
                self.qp.drawPoint(p2[0], p2[1])