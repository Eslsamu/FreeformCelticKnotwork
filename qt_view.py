#!/usr/bin/python

"""
ZetCode PyQt5 tutorial

In this example, we draw text in Russian Cylliric.

Author: Jan Bodnar
Website: zetcode.com
"""
import os
import sys
import subprocess
import dmsh
import dt
import cv2 as cv

from PyQt5.QtWidgets import QWidget, QPushButton,\
    QApplication, QFileDialog, QGridLayout,  \
    QSlider, QGroupBox, QVBoxLayout, QLabel, QMainWindow

from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint
from mercat import compute_curves
from mesh_init import init_pixels_go
import numpy as np
from scipy.interpolate import interp1d


"""
    Class inherited from a QWidget which represents our drawing area. 
    """

class ImageArea(QWidget):

    image = None

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        """
        Initializes two images which will be used later to resize or undo.
        """
        self.resizeSavedImage = QImage(0, 0, QImage.Format_RGB32)
        self.savedImage = QImage(0, 0, QImage.Format_RGB32)

        """
        Sets our default image with the right size filled in white.
        """
        self.background = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.background.fill(Qt.white)

        # init painter
        self.qp = QPainter()

    """
    Method called when the widget is resized.
    The image needs to be scaled with the new size or problems will occur.
    """
    def resizeEvent(self, event):
        self.background = self.background.scaled(self.width(), self.height())
        if self.image:
            self.image = self.image.scaled(self.width(), self.height())

    """
    Method called when a painting event occurs.
    """
    def paintEvent(self, event):
        self.qp.begin(self)
        self.qp.drawImage(self.rect(), self.background, self.background.rect())

        # draw the image
        if self.parent.img_file:
            self.image = QImage(self.parent.img_file)
            self.qp.drawImage(self.image.rect(), self.image)

        self.qp.end()

class MeshArea(QWidget):

    selected_vertex = None
    vertex_width = 10

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        """
        Initializes two images which will be used later to resize or undo.
        """
        self.resizeSavedImage = QImage(0, 0, QImage.Format_RGB32)
        self.savedImage = QImage(0, 0, QImage.Format_RGB32)

        """
        Sets our default image with the right size filled in white.
        """
        self.background = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.background.fill(Qt.white)

        # init painter
        self.qp = QPainter()
        self.black_pen = QPen(Qt.black, self.vertex_width / 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    """
    Method called when the widget is resized.
    The image needs to be scaled with the new size or problems will occur.
    """
    def resizeEvent(self, event):
        self.background = self.background.scaled(self.width(), self.height())

    """
    Method called when a painting event occurs.
    """
    def paintEvent(self, event):
        self.qp.begin(self)
        self.qp.drawImage(self.rect(), self.background, self.background.rect())

        #draw the graph
        if self.parent.display_graph:
            self.qp.setPen(self.black_pen)
            self.draw_graph()

        self.qp.end()

    def mousePressEvent(self, e):
        graph = self.parent.graph

        if e.button() == 1:
            clicked_vertex = [i for i, v in enumerate(graph["vertices"]) if
                              (e.x() - self.vertex_width < v[0] < e.x() + self.vertex_width) and
                              (e.y() - self.vertex_width < v[1] < e.y() + self.vertex_width)]
            if self.selected_vertex is not None:
                if len(clicked_vertex) == 1:
                    new_edge = (self.selected_vertex, clicked_vertex[0])
                    if new_edge not in graph["edges"] and new_edge[0] != new_edge[1]:
                        graph["edges"].append(new_edge)
                        print("added edge")
                else:
                    self.selected_vertex = None
                    print("deselect vertex")
            else:
                if len(clicked_vertex) == 1:
                    self.selected_vertex = clicked_vertex[0]
                    print("selected vertex", self.selected_vertex)
                else:
                    graph["vertices"].append([e.x(), e.y()])
                    print("add vertex", e.x(), e.y())
        if e.button() == 2:
            clicked_vertex = [i for i, v in enumerate(graph["vertices"]) if
                              (e.x() - self.vertex_width < v[0] < e.x() + self.vertex_width) and
                              (e.y() - self.vertex_width < v[1] < e.y() + self.vertex_width)]
            # delete vertex and adjacent edges
            if len(clicked_vertex) == 1:
                print("delete vertex", clicked_vertex[0])
                graph["vertices"].pop(clicked_vertex[0])
                graph["edges"] = [edge for edge in graph["edges"] if clicked_vertex[0] not in edge]

                # shift index by one after this vertex for all other edges
                for i, edge in enumerate(graph["edges"]):
                    v1, v2 = edge[0], edge[1]
                    if v1 > clicked_vertex[0]:
                        v1 = v1 - 1
                    if v2 > clicked_vertex[0]:
                        v2 = v2 - 1
                    graph["edges"][i] = (v1, v2)
                if self.selected_vertex:
                    if self.selected_vertex > clicked_vertex[0]:
                        self.selected_vertex = self.selected_vertex - 1
                    elif self.selected_vertex == clicked_vertex[0]:
                        print("y")
                        self.selected_vertex = None
            else:
                if self.selected_vertex:
                    graph["vertices"][self.selected_vertex] = (e.x(), e.y())
                    print("move vertex")

        self.update()

    def draw_graph(self):
        vertices = self.parent.graph["vertices"]
        edges = self.parent.graph["edges"]
        #draw vertices
        for v in vertices:
            # draw vertices
            x = v[0]
            y = v[1]
            self.qp.drawPoint(x,y)
        #draw edges
        for e in edges:
            #draw edge
            v1 = vertices[e[0]]
            v1 = QPoint(v1[0],v1[1])
            v2 = vertices[e[1]]
            v2 = QPoint(v2[0],v2[1])
            self.qp.drawLine(v1,v2)

    def draw_medial_graph(self,show_points=True):
        curves = compute_curves(self.parent.graph["vertices"], self.parent.graph["edges"])
        for p0, p1, p2, p3 in curves:
            path = QPainterPath(QPoint(p0[0],p0[1]))
            path.cubicTo(p1[0],p1[1], p2[0],p2[1], p3[0], p3[1])
            self.qp.drawPath(path)
            #TODO
            if show_points:
                self.qp.setPen(self.red_pen)
                self.qp.drawPoint(p1[0],p1[1])
                self.qp.drawPoint(p2[0],p2[1])
                self.qp.setPen(self.green_pen)


"""
Main class inherited from a QMainWindow which is the main window of the program.
"""
class Window(QMainWindow):

    vertex_width = 10
    display_medial_graph = False
    display_graph = True
    dmsh.generate = dt.generate
    img_file = None
    contour_points = []
    contour_level = 0

    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 1000)
        self.init_ui()
        self.graph = {"vertices":[],"edges":[]}

    def init_ui(self):
        self.grid = QGridLayout()

        # TODO menu

        #graphic areas
        self.imgArea = ImageArea(self)
        self.meshArea = MeshArea(self)
        self.grid.addWidget(self.imgArea, 0, 0, 2, 2)
        self.grid.addWidget(self.meshArea, 0, 2, 2, 2)

        # buttons and other input
        user_input = self.create_input()
        self.grid.addWidget(user_input, 2, 1, 1, 4)

        #central widget
        win = QWidget()
        win.setLayout(self.grid)
        self.setCentralWidget(win)

        self.update()

    def create_input(self):
        # box input elements together
        groupbox = QGroupBox()
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)

        # button 1
        self.button1 = QPushButton("load pic")
        self.button1.clicked.connect(self.load_pic)
        vbox.addWidget(self.button1)

        # button 2
        self.button2 = QPushButton("show medial graph")
        self.button2.clicked.connect(self.show_medial_graph)
        vbox.addWidget(self.button2)

        # button 3
        self.button3 = QPushButton("meshing")
        self.button3.clicked.connect(self.meshing)
        vbox.addWidget(self.button3)

        # button 4
        self.button4 = QPushButton("show graph")
        self.button4.clicked.connect(self.show_graph)
        vbox.addWidget(self.button4)

        #button 5
        self.button5 = QPushButton("cut mask")
        self.button5.clicked.connect(self.grabcut)
        vbox.addWidget(self.button5)

        #button 6
        self.button6 = QPushButton("find contours")
        self.button6.clicked.connect(self.findContours)
        vbox.addWidget(self.button6)

        # slider nominal distance
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(0)
        # slider value label
        self.label = QLabel()
        self.slider.valueChanged.connect(self.set_contour_level)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.label)

        return groupbox

    def set_contour_level(self):
        self.contour_level = self.slider.value()
        self.label.setText(str(self.contour_level))

    def show_medial_graph(self):
        self.display_medial_graph = not self.display_medial_graph
        if self.display_medial_graph:
            self.button3.setText("hide medial graph")
        else:
            self.button3.setText("show medial graph")
        self.imgArea.update()

    def show_graph(self):
        self.display_graph = not self.display_graph
        if self.display_graph:
            self.button4.setText("hide graph")
        else:
            self.button4.setText("show graph")
        self.meshArea.update()

    #a function that interpolates points based on the arc length of a curve
    #to create n regularly spaced points
    def regularize_contour(self, points, n):
        #independent vector for x and y
        x = points[:,0]
        y = points[:,1]

        #close the contour temporarily
        xc = np.append(x,x[0])
        yc = np.append(y,y[0])

        #distance between consecutive points
        dx = np.diff(xc)
        dy = np.diff(yc)
        dS = np.sqrt(dx**2+dy**2)
        dS = np.append(0,dS) #include starting point

        #arc length
        d = np.cumsum(dS)
        perim = d[-1]

        #create n segments
        ds = perim / n
        dSi = ds * np.arange(n)

        #interpolate
        xi = interp1d(d,xc)(dSi)[:-1]
        yi = interp1d(d,yc)(dSi)[:-1]

        return np.stack([xi,yi],axis=1)

    def findContours(self):
        if not self.img_file:
            self.imgArea.load_pic()

        im = cv.imread(self.img_file)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #algorithm that takes set of n points with minimal difference in distance to neighbors! -> makes contour points more regular
        points = cv.approxPolyDP(contours[self.contour_level], 1, False)
        self.contour_points = self.regularize_contour(points.reshape(len(points), 2),30)
        self.slider.setRange(0, len(contours))
        # update image to contain contours
        cv.drawContours(im, contours, self.contour_level, (0, 0, 255), 3)
        if self.img_file[0:5] != "cont_":
            self.img_file = "cont_" + os.path.basename(self.img_file)
        cv.imwrite(self.img_file, im)
        self.imgArea.update()

    def meshing(self):
        if self.contour_points is not None:
            self.findContours()

        geo = dmsh.Polygon(self.contour_points)
        X, _, edges = dmsh.generate(geo, 10)
        print(X.shape, edges.shape)

        #update graph
        self.graph["vertices"] = X.tolist()
        self.graph["edges"] = edges.tolist()
        self.meshArea.update()

    def load_pic(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.button1.setText(self.img_file)
        self.imgArea.update()

    def grabcut(self):
        subprocess.run(['python3 grabCut.py \"' + self.img_file + '\"'], shell=True)
        self.img_file = "mask_" + os.path.basename(self.img_file)
        self.imgArea.update()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()