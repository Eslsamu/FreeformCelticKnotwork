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
import meshplex

from PyQt5.QtWidgets import QWidget, QPushButton,\
    QApplication, QFileDialog, QGridLayout,  \
    QSlider, QGroupBox, QVBoxLayout, QLabel, QMainWindow, QScrollArea

from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage, QPolygon, QColor, QBrush, QPalette
from PyQt5.QtCore import Qt, QPoint
from mercat import compute_curves

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from shapely.ops import polylabel
from shapely.geometry import Polygon, Point


"""
    Class inherited from a QWidget which represents our drawing area. 
    """

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

class MeshArea(QWidget):

    selected_vertex = None
    vertex_width = 10

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
        self.black_pen = QPen(Qt.black, self.vertex_width / 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.vertex_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

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


        #mark the contour points
        if self.parent.display_contour_points:
            for x, y in self.parent.contour_points[self.parent.selected_segment]:
                self.qp.setPen(self.green_pen)
                self.qp.drawPoint(x,y)

        #adjust widget size based on loaded image
        img = self.parent.imgArea.image
        if img is not None:
            self.setMinimumWidth(img.width()*1.2)
            self.setMinimumHeight(img.height() * 1.2)


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
        self.draw_mesh()
        """
        
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
            """

    def draw_mesh(self):
        if self.parent.mesh is not None:
            mesh = self.parent.mesh
            coords = mesh.node_coords
            for i,n in enumerate(mesh.cells["nodes"]):
                a = QPoint(coords[n[0]][0],coords[n[0]][1])
                b = QPoint(coords[n[1]][0],coords[n[1]][1])
                c = QPoint(coords[n[2]][0],coords[n[2]][1])

                #draw the cell
                path = QPainterPath()
                path.moveTo(a)
                path.lineTo(b)
                path.lineTo(c)
                path.closeSubpath()
                path.moveTo(a)

                #fill the cell with color intensity based on the cell quality
                qual = mesh.cell_quality[i]
                self.qp.setBrush(QBrush(QColor(255,0,0,255-int(qual*255))))
                self.qp.drawPath(path)

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

"""
Main class inherited from a QMainWindow which is the main window of the program.
"""
class Window(QMainWindow):

    knot_width = 10
    vertex_width = 10
    display_knot = True
    display_graph = True
    display_contour_points = False
    display_control_points = False
    dmsh.generate = dt.generate
    processed_img = None
    img_file = None
    display_img_file = None
    contour_points = []
    segments = []
    selected_segment = 0
    contour_roughness = 0.05
    blur = 2
    canny_th = 0.2
    min_edge_size = 15
    mesh = None

    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 1000)
        self.init_ui()
        #TODO mesh into graph
        self.graph = {"vertices":[],"edges":[]}

    def init_ui(self):
        self.grid = QGridLayout()

        # TODO menu

        #graphic areas
        self.imgArea = ImageArea(self)
        self.meshArea = MeshArea(self)
        self.knotArea = KnotArea(self)

        #display the areas inside a scrolling view
        self.scrollArea1 = QScrollArea()
        self.scrollArea1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea1.setWidgetResizable(True)
        self.scrollArea1.setWidget(self.imgArea)

        # display the areas inside a scrolling view
        self.scrollArea2 = QScrollArea()
        self.scrollArea2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea2.setWidgetResizable(True)
        self.scrollArea2.setWidget(self.meshArea)

        # display the areas inside a scrolling view
        self.scrollArea3 = QScrollArea()
        self.scrollArea3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea3.setWidgetResizable(True)
        self.scrollArea3.setWidget(self.knotArea)

        self.grid.addWidget(self.scrollArea1, 0, 0, 2, 2)
        self.grid.addWidget(self.scrollArea2, 0, 2, 2, 2)
        self.grid.addWidget(self.scrollArea3, 0, 4, 2, 2)

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
        grid = QGridLayout()
        groupbox.setLayout(grid)

        # button 1
        self.button1 = QPushButton("load pic")
        self.button1.clicked.connect(self.load_pic)
        grid.addWidget(self.button1, 0, 0)

        # button 2
        self.button2 = QPushButton("show knot")
        self.button2.clicked.connect(self.show_knot)
        grid.addWidget(self.button2, 1, 0)

        # button 3
        self.button3 = QPushButton("meshing")
        self.button3.clicked.connect(self.meshing)
        grid.addWidget(self.button3, 2,0)

        # button 4
        self.button4 = QPushButton("show graph")
        self.button4.clicked.connect(self.show_graph)
        grid.addWidget(self.button4, 3, 0)

        #button 5
        self.button5 = QPushButton("cut mask")
        self.button5.clicked.connect(self.grabcut)
        grid.addWidget(self.button5, 0, 1)

        #button 6
        self.button6 = QPushButton("find contours")
        self.button6.clicked.connect(self.findContours)
        grid.addWidget(self.button6, 1,1)

        # button 7
        self.button7 = QPushButton("show contour points")
        self.button7.clicked.connect(self.show_contour_points)
        grid.addWidget(self.button7,2,1)

        # slider contour level
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 10)
        self.slider1.setValue(self.selected_segment)
        # slider value label
        self.label1 = QLabel()
        self.slider1.valueChanged.connect(self.set_selected_segment)
        self.set_selected_segment()
        grid.addWidget(self.slider1,0,2)
        grid.addWidget(self.label1,0,3)

        # slider edge size
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(5, 100)
        self.slider2.setValue(self.min_edge_size)
        # slider value label
        self.label2 = QLabel()
        self.slider2.valueChanged.connect(self.set_min_edge_size)
        self.set_min_edge_size()
        grid.addWidget(self.slider2,1,2)
        grid.addWidget(self.label2,1,3)

        # slider knot width
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(1, 30)
        self.slider3.setValue(self.knot_width)
        # slider value label
        self.label3 = QLabel()
        self.slider3.valueChanged.connect(self.set_knot_width)
        self.set_knot_width()
        grid.addWidget(self.slider3,2,2)
        grid.addWidget(self.label3,2,3)

        # slider contour roughness
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setRange(0, 20)
        self.slider4.setSingleStep(1)
        self.slider4.setValue(self.contour_roughness*100)
        # slider value label
        self.label4 = QLabel()
        self.slider4.valueChanged.connect(self.set_contour_roughness)
        self.set_contour_roughness()
        grid.addWidget(self.slider4,3,2)
        grid.addWidget(self.label4,3,3)

        # slider blur
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setRange(0, 10)
        self.slider5.setSingleStep(1)
        self.slider5.setValue(self.blur)
        # slider value label
        self.label5 = QLabel()
        self.slider5.valueChanged.connect(self.set_blur)
        self.set_blur()
        grid.addWidget(self.slider5, 4, 2)
        grid.addWidget(self.label5, 4, 3)

        return groupbox

    def set_selected_segment(self):
        self.selected_segment = self.slider1.value()
        self.label1.setText("contour lvl " + str(self.selected_segment))
        self.contour_points = []

        if self.processed_img is not None:
            self.draw_segments()

    def set_min_edge_size(self):
        self.min_edge_size = self.slider2.value()
        self.label2.setText("edge size " + str(self.min_edge_size))

    def set_knot_width(self):
        self.knot_width = self.slider3.value()
        self.label3.setText("knot width " + str(self.knot_width))
        self.knotArea.update()

    def set_contour_roughness(self):
        self.contour_roughness = self.slider4.value() / 100
        self.label4.setText("contour roughness " + str(self.contour_roughness))
        self.contour_points = []

    def set_blur(self):
        self.blur = self.slider5.value()
        self.label5.setText("blur " + str(self.blur))
        self.contour_points = []

    def show_knot(self):
        self.display_knot = not self.display_knot
        if self.display_knot:
            self.button2.setText("hide knot")
        else:
            self.button2.setText("show knot")
        self.knotArea.update()

    def show_graph(self):
        self.display_graph = not self.display_graph
        if self.display_graph:
            self.button4.setText("hide graph")
        else:
            self.button4.setText("show graph")
        self.meshArea.update()

    def show_contour_points(self):
        self.display_contour_points = not self.display_contour_points
        if self.display_contour_points:
            self.button7.setText("hide contour points")
        else:
            self.button7.setText("show contour points")
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
            self.load_pic()

        im = cv.imread(self.img_file)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        median = np.median(imgray)
        thresh = median * self.canny_th
        edges = cv.Canny(im, thresh, thresh * 2)

        #TODO
        if self.blur > 0:
            edges = cv.blur(edges,(self.blur,self.blur))



        self.contour_points, h = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        print("inital amount of cnts", len(self.contour_points))

        #remove all contours with arclength less than 3 * edge length
        cnts = []
        for contour in self.contour_points:
            if cv.arcLength(contour,True) > self.min_edge_size * 3:
                cnts.append(contour)
        print("removed", len(self.contour_points) - len(cnts), "short contours")
        self.contour_points = cnts

        #remove the inner part of a contours (hole)
        #in opencv they are represented with a negative sign for their area --> not working correctly most of the time
        """
        cnts = []
        for contour in self.contour_points:
            if cv.contourArea(contour, True) > 0:
                cnts.append(contour)
            else:
                print("remove inner part")
        self.contour_points = cnts
        """

        #TODO till wednesday night

        #TODO graph mesh display

        #TODO is polylabel working?
        #TODO apaptive edge length and minimum contour

        #TODO eyeball case (color/gray blur problem) --> maybe use edges for rgb and gray
        #TODO smartly remove details from images with k-means or watershed (like viking face, viking hammer)

        #TODO replace bad quality cells with single graph elements

        # delete contours with less than 3 points (not even forming a polygon)
        cnts = []
        for contour in self.contour_points:
            if len(contour) > 2:
                cnts.append(contour)
        print("removed", len(self.contour_points) - len(cnts), " contours with less than 3 points")
        self.contour_points = cnts


        #simplify polygons using a topology preserving variant of the Visvalingam-Whyatt Algorithm
        from simplification.cutil import (
            simplify_coords_vwp,
        )
        self.contour_points = [simplify_coords_vwp(np.squeeze(contour), 30.0) for contour in self.contour_points]

        # turn contours into polygon objects
        self.contour_points = [Polygon(contour) for contour in self.contour_points]

        cnts = []
        self.invalid_segments = []
        for polygon in self.contour_points:
            if polygon.is_valid:
                cnts.append(polygon)
            else:
                self.invalid_segments.append(polygon)
        self.contour_points = cnts


        #form segments from contours
        self.segments, self.thin_segments = self.contours_to_segments(self.contour_points)

        #remove too thin segments (distance of visual center to border is less than minimum edge size)
        #self.contour_points, self.thin_segments = self.remove_thin_segments(self.contour_points)


        print("found",len(self.contour_points),"valid contours")

        # set slider range for amount of segments
        self.slider1.setRange(0, len(self.segments) - 1)

        #save image
        self.processed_img = edges

        #draw segments
        self.draw_segments()

    def draw_segments(self):
        #grayscale image back into RGB to draw contours
        plt.imshow(np.array(self.processed_img))

        # draw all too thin segments
        for polygon in self.thin_segments:
            plt.plot(*polygon.exterior.xy,color = "blue",linewidth = 1)


        # draw all invalid contours
        for polygon in self.invalid_segments:
            plt.plot(*polygon.exterior.xy, color = "yellow",linewidth = 1)

        #draw segments
        if len(self.segments) > 0:
            for polygon in self.segments:
                # draw all approximated contours except the selected level on image as green lines
                if polygon != self.segments[self.selected_segment]:
                    plt.plot(*polygon.exterior.xy, color = "green",linewidth = 1)
                # draw approximated selected contour on image as red lines

            polygon = self.segments[self.selected_segment]
            plt.plot(*polygon.exterior.xy, color = "red",linewidth = 2)
            for int in polygon.interiors:
                plt.plot(*int.xy, color="red",linewidth = 2)
            try:
                label = polylabel(polygon, tolerance=1)
                plt.plot(label.x, label.y, color="red", linewidth = 5, marker="X")
            except Exception:
                print(polygon, self.selected_segment)

        #contour image file
        self.display_img_file = "cont_" + os.path.basename(self.img_file)
        plt.savefig(self.display_img_file)
        plt.close()
        self.imgArea.update()



    #a segment is represented as a polygon
    def contours_to_segments(self, contours):
        segments = []
        thin_segments = []
        for i in range(len(contours)):
            contour1 = contours[i]

            #find all contours inside inside the current contour
            contains = []
            for j in range(len(contours)):
                if contour1.contains(contours[j]) and i != j:
                    contains.append(contours[j])

            #create a segment of the current contour with its containing contours as holes
            segment = contour1
            for j, cnt in enumerate(contains):
                segment = segment.difference(cnt)

            # find visual center (polylabel) of segment
            label = polylabel(segment, tolerance=1)
            # determine distance from segment border
            dist = segment.exterior.distance(Point(label.x, label.y))
            # remove segment if the distance from its center is less than an edge size (too thin)
            if dist < self.min_edge_size:
                # print("too small contour", i, contour)
                thin_segments.append(segment)
            else:
                segments.append(segment)

        return segments, thin_segments


    def meshing(self):
        if len(self.segments) == 0:
            self.findContours()

        #TODO mesh each segment with adaptive edge size
        for i,segment in enumerate(self.segments):
            print("meshing segment",i,"/",len(self.segments))
            #exterior boundary (dmsh does not include the last point twice to close the polygon such as shapely)
            e = list(segment.exterior.coords.xy)
            outer = dmsh.Polygon([[e[0][i],e[1][i]] for i in range(len(e[0])-1)])

            #interior boundary
            inner = []
            for int in segment.interiors:
                h = list(int.coords.xy)
                hole = dmsh.Polygon([[h[0][i], h[1][i]] for i in range(len(h[0])-1)])
                inner.append(hole)

            if len(inner) > 1:
                inner = dmsh.Union(inner)
            elif inner == 1:
                inner = inner[0]
            else:
                pass

            geo = dmsh.Difference(outer,inner)

            #inspect segment
            plt.figure(figsize=(8, 8))
            plt.axis('equal')
            plt.plot(*segment.exterior.xy)
            for int in segment.interiors:
                plt.plot(*int.xy, color="red")
            plt.savefig(str(i) + "_shape.jpg")
            plt.close()

            #inspect geo
            geo.save(str(i)+"_geo.jpg")

            X, cells, edges = dmsh.generate(geo, edge_size=self.min_edge_size)

            #try to further optimize the mesh
            try:
                #optimize it with centroidal voronoi tesselation
                self.mesh = dt.quasi_newton_uniform_full(X, cells, 1.0e-10, 100)

                # save as image

                self.mesh.save(
                    str(i)+"_mesh.png", show_coedges=False, show_axes=False, nondelaunay_edge_color="k",
                )

            except meshplex.exceptions.MeshplexError as err:
                print(err)


        #TODO draw path and define distance function to path or figure out how this could be used automatically to make better knotworks
        #TODO e.g make distance function according to inner contours (internal features)

        #update graph
        self.graph["vertices"] = self.mesh.node_coords.tolist()
        self.graph["edges"] = self.mesh.edges["nodes"].tolist()
        self.meshArea.update()
        self.knotArea.update()



    """
    
    """
    def merge_border_cells(self, mesh, th=0.8):
        while True:
            #select border cells
            border_cells = []
            for i, cell in enumerate(mesh.cells["nodes"]):
                border_nodes_index = []
                border_nodes_coords = []
                for node in cell:
                    #check which nodes of cell touch the border
                    if mesh.node_coords[node] in self.contour_points:
                        border_nodes_index.append(node)
                        border_nodes_coords.append(mesh.node_coords[node])

                if len(border_nodes_index) > 0:
                    #store as [cell, cell index, [border node index], [border node coords], cell_quality]
                    border_cells.append([cell,i,border_nodes_index, border_nodes_coords,mesh.cell_quality[i]])
            border_cells = np.array(border_cells, dtype=object)

            #stop when the cell quality of all border cells is bigger than the treshold
            if np.min(border_cells[:,4]) > th:
                break

            #find the cell with lowest quality
            min_qual_i = np.argmin(border_cells[:,4])
            min_qual_cell = border_cells[min_qual_i]

            #find neighbor border cells
            neighbors = []
            for i in min_qual_cell[2]:
                for cell in border_cells:
                    if i in cell[2] and cell not in neighbors:
                        neighbors.append(cell)

            neighbors = np.array(neighbors)

            #option 1
            #merge remove a border node with low quality and then remesh --> super slow

            #option 2
            #merge with the neighbor that creates the highest quality cell

            #option 3
            #merge with the one that creates the least distortion in shape (based on knotwork)

            #option 4
            #remove the border node that results in the lowest quality based on
            # for each cell it belongs to / cells it belongs to
            #then retriangulate
            #or even reoptimize
            """
            #for point in min_qual_cell[2]:
            leaving_border_node = min_qual_cell[3][0]
            X = np.array([node for node in mesh.node_coords if (node != leaving_border_node).any()])
            cells = np.array([cell for cell in mesh.cells["nodes"] if (cell != min_qual_cell[0]).all()])
            print("shape old new X",mesh.node_coords.shape, X.shape)
            print("shape old new cells", mesh.cells["nodes"].shape,cells.shape)
            #shift index
            for i,cell in enumerate(cells):
                for j,node in enumerate(cell):
                    if node > min_qual_cell[2][0]: #index of leaving border node
                        cells[i][j] -= 1

            mesh = dt.quasi_newton_uniform_full(X, cells, 1.0e-10, 100)
            """

            self.mesh = mesh
            self.meshArea.update()
            break

    """
            remove contour points that have the smallest inflection 
            """
    def remove_small_inflection_points(self):

        points = self.contour_points.tolist()

        while True:
            if len(points) < 3:
                break
            length = len(points)

            i = 0
            while i < len(points):
                print(i, len(points))
                # compute angle between three points, i.e degree of inflection for the midpoint
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                p3 = points[(i + 2) % len(points)]
                p12 = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                p13 = np.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
                p23 = np.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
                angle = np.arccos((p12 ** 2 + p13 ** 2 - p23 ** 2) / (2 * p12 * p13))
                print(p1, p2, p3, angle)
                print(p12, p23)
                if angle < self.contour_roughness and p12 < self.min_edge_size and p23 < self.min_edge_size:
                    print("remove", p2)
                    points.pop((i + 1) % len(points))
                elif angle < self.contour_roughness:
                    # if the inflection is high but the distance between the points is too small
                    # TODO save the point to add a single vertex to the graph later (and maybe merge the two neighbor points)
                    pass
                i += 1
            if length == len(points):
                break

        self.contour_points = np.array(points)

    def load_pic(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.display_img_file = self.img_file
        self.button1.setText(os.path.basename(self.img_file))
        self.contour_points = []
        self.imgArea.update()

    def grabcut(self):
        subprocess.run(['python3 grabCut.py \"' + self.img_file + '\"'], shell=True)
        self.img_file = "mask_" + os.path.basename(self.img_file)
        self.display_img_file = self.img_file
        self.imgArea.update()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()