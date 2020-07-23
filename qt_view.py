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
    QApplication, QFileDialog, QGridLayout, QInputDialog, \
    QSlider, QGroupBox, QHBoxLayout,QVBoxLayout, QLabel

from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint
from mercat import compute_curves
from mesh_init import init_pixels_go
import numpy as np
import json

class MeshUI(QWidget):

    vertex_width = 10
    selected_vertex = None
    display_medial_graph = False
    display_graph = True
    img_file = None
    nominal_distance = 100
    dmsh.generate = dt.generate

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.graph = {"vertices":[],"edges":[]}

        #init painter
        self.qp = QPainter()
        self.black_pen = QPen(Qt.black, self.vertex_width/2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def init_ui(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setGeometry(0, 0, 1000, 1000)

        #TODO menu

        #buttons and other input
        user_input = self.create_input()
        self.layout.addWidget(user_input,1,1,Qt.AlignBottom)

        #graphics
        graphics = self.create_graphics()
        self.layout.addWidget(graphics,1,2,Qt.AlignTop)

        self.show()

    def create_graphics(self):
        # box input elements together
        groupbox = QGroupBox()
        hbox = QHBoxLayout()
        groupbox.setLayout(hbox)

        #TODO

        # image
        self.image_label = QLabel("image")
        hbox.addWidget(self.image_label)

        # meshing
        self.mesh_label = QLabel("mesh")
        hbox.addWidget(self.mesh_label)

        # knotwork
        self.knotwork_label = QLabel("knot")
        hbox.addWidget(self.knotwork_label)

        return groupbox

    def create_input(self):
        #box input elements together
        groupbox = QGroupBox()
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)

        #button 1
        self.button1 = QPushButton("init lattice")
        self.button1.clicked.connect(self.init_lattice)
        vbox.addWidget(self.button1)

        #button 2
        self.button2 = QPushButton("load pic")
        self.button2.clicked.connect(self.load_pic)
        vbox.addWidget(self.button2)

        #button 3
        self.button3 = QPushButton("show medial graph")
        self.button3.clicked.connect(self.show_medial_graph)
        vbox.addWidget(self.button3)

        #button 4
        self.button4 = QPushButton("meshing")
        self.button4.clicked.connect(self.meshing)
        vbox.addWidget(self.button4)

        #button 5
        self.button5 = QPushButton("show graph")
        self.button5.clicked.connect(self.show_graph)
        vbox.addWidget(self.button5)

        #slider nominal distance
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1,200)
        self.slider.setValue(100)
        #slider value label
        self.label = QLabel()
        self.slider.valueChanged.connect(self.set_nominal_distance)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.label)

        return groupbox

    def set_nominal_distance(self):
        self.nominal_distance = self.slider.value()
        self.label.setText(str(self.nominal_distance))

    def init_lattice(self):
        if self.img_file:
            lattice = init_pixels_go(self.img_file, amount = 20)
            for v in lattice:
                self.graph["vertices"].append(v)
        self.update()

    def show_medial_graph(self):
        self.display_medial_graph = not self.display_medial_graph
        if self.display_medial_graph:
            self.button3.setText("hide medial graph")
        else:
            self.button3.setText("show medial graph")
        self.update()

    def show_graph(self):
        self.display_graph = not self.display_graph
        if self.display_graph:
            self.button5.setText("hide graph")
        else:
            self.button5.setText("show graph")
        self.update()

    def mousePressEvent(self, e):
        if e.button() == 1:
            clicked_vertex = [i for i, v in enumerate(self.graph["vertices"]) if
                              (e.x() - self.vertex_width < v[0] < e.x() + self.vertex_width) and
                              (e.y() - self.vertex_width < v[1] < e.y() + self.vertex_width)]
            if self.selected_vertex is not None:
                if len(clicked_vertex) == 1:
                    new_edge = (self.selected_vertex, clicked_vertex[0])
                    if new_edge not in self.graph["edges"] and new_edge[0] != new_edge[1]:
                        self.graph["edges"].append(new_edge)
                        print("added edge")
                else:
                    self.selected_vertex = None
                    print("deselect vertex")
            else:
                if len(clicked_vertex) == 1:
                    self.selected_vertex = clicked_vertex[0]
                    print("selected vertex", self.selected_vertex)
                else:
                    self.graph["vertices"].append([e.x(), e.y()])
                    print("add vertex", e.x(), e.y())
        if e.button() == 2:
            clicked_vertex = [i for i, v in enumerate(self.graph["vertices"]) if
                              (e.x() - self.vertex_width < v[0] < e.x() + self.vertex_width) and
                              (e.y() - self.vertex_width < v[1] < e.y() + self.vertex_width)]
            # delete vertex and adjacent edges
            if len(clicked_vertex) == 1:
                print("delete vertex", clicked_vertex[0])
                self.graph["vertices"].pop(clicked_vertex[0])
                self.graph["edges"] = [edge for edge in self.graph["edges"] if clicked_vertex[0] not in edge]

                # shift index by one after this vertex for all other edges
                for i, edge in enumerate(self.graph["edges"]):
                    v1, v2 = edge[0], edge[1]
                    if v1 > clicked_vertex[0]:
                        v1 = v1 - 1
                    if v2 > clicked_vertex[0]:
                        v2 = v2 - 1
                    self.graph["edges"][i] = (v1, v2)
                if self.selected_vertex:
                    if self.selected_vertex > clicked_vertex[0]:
                        self.selected_vertex = self.selected_vertex - 1
                    elif self.selected_vertex == clicked_vertex[0]:
                        print("y")
                        self.selected_vertex = None
            else:
                if self.selected_vertex:
                    self.graph["vertices"][self.selected_vertex] = (e.x(), e.y())
                    print("move vertex")

        self.update()

    def paintEvent(self,e):
        self.qp.begin(self)

        #first draw the image
        if self.img_file:
            self.img = QImage(self.img_file)
            self.qp.drawImage(self.img.rect(),self.img)

        #draw the graph
        if self.display_graph:
            self.qp.setPen(self.black_pen)
            self.draw_graph()

        #highlight selected vertex
        if self.selected_vertex:
            self.qp.setPen(self.red_pen)
            vertex = self.graph["vertices"][self.selected_vertex]
            self.qp.drawPoint(QPoint(vertex[0],vertex[1]))

        #draw medial graph
        if self.display_medial_graph:
            self.qp.setPen(self.green_pen)
            self.draw_medial_graph()

        self.qp.end()

    def meshing2(self):
        iterations = QInputDialog.getInt(self,"","iterations",0,0,100000)[0]
        beta = QInputDialog.getDouble(self,"","beta",0.5,0,1)[0]

        msg = {"Atoms": ' '.join(str(x) for x in self.graph["vertices"]),
               "Iterations": iterations,
               "NominalDistance": self.nominal_distance,
               "Beta": beta,
               "Image": self.img_file
               }

        msg = json.dumps(msg)
        p = subprocess.run(['go run atomic_meshing.go'],
                                 shell=True, stdout=subprocess.PIPE,
                               input=msg, encoding='ascii')
        output = p.stdout
        try:
            result = json.loads(output)
            print("result", result)
            print("pre",self.graph["vertices"])
            self.graph['vertices'] = np.reshape(result["X"],(int(len(result["X"])/2),2)).tolist()
            #print("post",self.graph)
            self.update()
        except json.decoder.JSONDecodeError:
            print(output)
            print("pre",self.graph["vertices"])

    def meshing(self):
        if not self.img_file:
            self.load_pic()

        im = cv.imread(self.img_file)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        points = cv.approxPolyDP(contours[0], 1, False)
        points = points.reshape(len(points), 2)
        print(points.shape)

        geo = dmsh.Polygon(points)
        X, _, edges = dmsh.generate(geo, 50)
        print(X.shape, edges.shape)
        self.graph["vertices"] = X.tolist()
        self.graph["edges"] = edges.tolist()
        self.update()

    def load_pic(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.button2.setText(self.img_file)
        subprocess.run(['python3 grabCut.py \"'+self.img_file+'\"'],shell=True)
        self.img_file = "mask_" + os.path.basename(self.img_file)

        self.update()

    def draw_graph(self):
        vertices = self.graph["vertices"]
        edges = self.graph["edges"]
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
        curves = compute_curves(self.graph["vertices"], self.graph["edges"])
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


def main():
    app = QApplication(sys.argv)
    ui = MeshUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()