#!/usr/bin/python

"""
ZetCode PyQt5 tutorial

In this example, we draw text in Russian Cylliric.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
import subprocess
import time

from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QFileDialog, QGridLayout, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint
from mercat import compute_curves
from mesh_init import init_pixels_go
import numpy as np
import json

class MeshUI(QWidget):

    vertex_width = 10
    selected_vertex = None
    medial_graph = None
    img_file = "x.png"#None

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.graph = {"vertices":[],"edges":[]}

        #init painter
        self.qp = QPainter()
        self.black_pen = QPen(Qt.black, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        self.init_lattice()
        self.meshing()


    def init_ui(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setGeometry(0, 0, 1000, 1000)

        #button 1
        self.button1 = QPushButton("init lattice")
        self.button1.clicked.connect(self.init_lattice)
        self.layout.addWidget(self.button1,1,1,Qt.AlignBottom)

        #button 2
        self.button2 = QPushButton("load pic")
        self.button2.clicked.connect(self.load_pic)
        self.layout.addWidget(self.button2,1,2,Qt.AlignBottom)

        #button 3
        self.button3 = QPushButton("show medial graph")
        self.button3.clicked.connect(self.show_medial_graph)
        self.layout.addWidget(self.button3,1,3,Qt.AlignBottom)

        #button 4
        self.button4 = QPushButton("meshing")
        self.button4.clicked.connect(self.meshing)
        self.layout.addWidget(self.button4,1,4,Qt.AlignBottom)

        self.show()

    def init_lattice(self):
        if self.img_file:
            lattice = init_pixels_go(self.img_file)
            for v in lattice:
                self.graph["vertices"].append(v)
        self.update()

    def show_medial_graph(self):
        self.medial_graph = not self.medial_graph
        if self.medial_graph:
            self.button3.setText("hide medial graph")
        else:
            self.button3.setText("show medial graph")
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
                    print("add vertex")

            self.update()

    def paintEvent(self,e):
        self.qp.begin(self)

        #first draw the image
        if self.img_file:
            img = QImage(self.img_file)
            self.qp.drawImage(img.rect(),img)
            print()

        #draw the graph
        self.qp.setPen(self.black_pen)
        self.draw_graph()

        #highlight selected vertex
        if self.selected_vertex:
            self.qp.setPen(self.red_pen)
            vertex = self.graph["vertices"][self.selected_vertex]
            self.qp.drawPoint(QPoint(vertex[0],vertex[1]))

        #draw medial graph
        if self.medial_graph:
            self.qp.setPen(self.black_pen)
            self.draw_medial_graph()

        self.qp.end()

    def meshing(self):
        #TODO iterations
        iterations = QInputDialog.getInt(self,"","iterations",0,0,100000)[0]
        d = 0
        beta = 0

        msg = {"Atoms": ' '.join(str(x) for x in self.graph["vertices"]),
               "Iterations": iterations,
               "NominalDistance": d,
               "Beta": beta
               }

        msg = json.dumps(msg)

        p = subprocess.run(['go run atomic_meshing.go'],
                                 shell=True, stdout=subprocess.PIPE,
                               input=msg, encoding='ascii')
        output = p.stdout
        result = json.loads(output)
        print("result", result)
        print("pre",self.graph["vertices"])
        self.graph['vertices'] = np.reshape(result["X"],(len(self.graph["vertices"]),2)).tolist()
        print("post",self.graph)
        self.update()

    def load_pic(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.button2.setText(self.img_file)
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
                self.qp.setPen(self.black_pen)


def main():
    app = QApplication(sys.argv)
    ui = MeshUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()