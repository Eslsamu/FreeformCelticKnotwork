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
from mesh_init import init_pixels_go
import numpy as np
from scipy.interpolate import interp1d
from shapely.ops import polylabel
from shapely.geometry import Polygon


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
            for x, y in self.parent.contour_points[self.parent.contour_level]:
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
    img_file = None
    display_img_file = None
    contour_points = []
    contour_level = 0
    contour_roughness = 0.01
    edge_size = 70
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
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)

        # button 1
        self.button1 = QPushButton("load pic")
        self.button1.clicked.connect(self.load_pic)
        vbox.addWidget(self.button1)

        # button 2
        self.button2 = QPushButton("show knot")
        self.button2.clicked.connect(self.show_knot)
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

        # button 7
        self.button7 = QPushButton("show contour points")
        self.button7.clicked.connect(self.show_contour_points)
        vbox.addWidget(self.button7)

        # button 8
        self.button8 = QPushButton("test shape")
        self.button8.clicked.connect(self.test_shape)
        vbox.addWidget(self.button8)


        # slider contour level
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 10)
        self.slider1.setValue(self.contour_level)
        # slider value label
        self.label1 = QLabel()
        self.slider1.valueChanged.connect(self.set_contour_level)
        vbox.addWidget(self.slider1)
        vbox.addWidget(self.label1)

        # slider edge length
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(5, 100)
        self.slider2.setValue(self.edge_size)
        # slider value label
        self.label2 = QLabel()
        self.slider2.valueChanged.connect(self.set_edge_size)
        vbox.addWidget(self.slider2)
        vbox.addWidget(self.label2)

        # slider knot width
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(1, 30)
        self.slider3.setValue(self.knot_width)
        # slider value label
        self.label3 = QLabel()
        self.slider3.valueChanged.connect(self.set_knot_width)
        vbox.addWidget(self.slider3)
        vbox.addWidget(self.label3)

        # slider contour roughness
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setRange(0, 20)
        self.slider4.setSingleStep(1)
        self.slider4.setValue(self.contour_roughness*1000)
        # slider value label
        self.label4 = QLabel()
        self.slider4.valueChanged.connect(self.set_contour_roughness)
        vbox.addWidget(self.slider4)
        vbox.addWidget(self.label4)

        return groupbox

    def set_contour_level(self):
        self.contour_level = self.slider1.value()
        self.label1.setText("contour lvl " + str(self.contour_level))
        self.contour_points = []

    def set_edge_size(self):
        self.edge_size = self.slider2.value()
        self.label2.setText("edge size " + str(self.edge_size))

    def set_knot_width(self):
        self.knot_width = self.slider3.value()
        self.label3.setText("knot width " + str(self.knot_width))
        self.knotArea.update()

    def set_contour_roughness(self):
        self.contour_roughness = self.slider4.value() / 1000
        self.label4.setText("contour roughness " + str(self.contour_roughness))
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
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        all_contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        #remove all contours with arclength less than 3 * edge length
        self.contour_points = []
        for contour in all_contours:
            if cv.arcLength(contour,True) > self.edge_size * 3:
                self.contour_points.append(contour)

        #make contour segments and by making that remove contours which result in a too small area segment

        #approximate contours by ramer douglas peucker algorithm
        self.contour_points = [cv.approxPolyDP(contour,
                 self.contour_roughness*cv.arcLength(self.contour_points[i],True), False).reshape(-1,2)
                 for i,contour in enumerate(self.contour_points)]

        #merge points that are very close together (less than 3/4 of one edge length)
        #necessary?

        #delete contours with less than 3 points (not even forming a polygon)
        for i, contour in enumerate(self.contour_points):
            if len(contour) < 3:
                del self.contour_points[i]


        #remove too thin segments
        self.contour_points = self.remove_thin_segments(self.contour_points)

        print("now got contours:", len(self.contour_points))
        #self.remove_small_inflection_points()

        self.slider1.setRange(0, len(self.contour_points)-1)

        cont_im = im.copy()
        # draw all approximated contours except the selected level on image as green lines
        for lvl in range(len(self.contour_points)):
            if lvl != self.contour_level:
                contour = self.contour_points[lvl].reshape(-1, 2)
                for i in range(len(contour)):
                    cv.line(cont_im, tuple(contour[i]), tuple(contour[(i+1) % len(contour)]), (0, 255, 0))

        # draw approximated selected contour on image as red lines
        contour = self.contour_points[self.contour_level].reshape(-1, 2)
        for i in range(len(contour)):
            cv.line(cont_im, tuple(contour[i]), tuple(contour[(i + 1) % len(contour)]), (0, 0, 255), thickness=2)

        #draw centroid of selected contour
        polygon = Polygon(np.squeeze(self.contour_points[self.contour_level]))
        try:
            label = polylabel(polygon, tolerance=1)
        except Exception:
            print(polygon, self.contour_level)
        cv.circle(cont_im, (int(label.x), int(label.y)), 5, (255, 0, 0), -1)

        #contour image file
        self.display_img_file = "cont_" + os.path.basename(self.img_file)
        cv.imwrite(self.display_img_file, cont_im)

        self.imgArea.update()

    # remove all contours with a too thin area
    # check if the distance from the polylabel of the contour to itself is less than 3/4 of its edge length
    # use opencv pointpolygontest
    # (which means we won't need to mesh in this area)
    # but maybe this solves itself but replacing small angle cells with single graph elements
    #TODO from contours to segments
    def remove_thin_segments(self, contours):
        new_contours = []
        for i, contour in enumerate(contours):
            # find visual center (polylabel) of segment
            polygon = Polygon(np.squeeze(contour))
            if not polygon.is_valid:
                print("invalid contour", i, contour)
                continue
            label = polylabel(polygon, tolerance=1)

            # determine distance from segment border
            dist = cv.pointPolygonTest(contour, (label.x, label.y), True)

            # remove segment if the distance from its center is less than an edge size (too small)
            if dist < self.edge_size:
                print("too small contour", i, contour)
            else:
                print("append", i, polygon)
                new_contours.append(contour)

        return new_contours

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
                print(p12,p23)
                if angle < self.contour_roughness and p12 < self.edge_size and p23 < self.edge_size:
                    print("remove", p2)
                    points.pop((i + 1) % len(points))
                elif angle < self.contour_roughness:
                    #if the inflection is high but the distance between the points is too small
                    #TODO save the point to add a single vertex to the graph later (and maybe merge the two neighbor points)
                    pass
                i += 1
            if length == len(points):
                break

        self.contour_points = np.array(points)

    #TODO remove small angle points with

    def meshing(self):
        if len(self.contour_points) == 0:
            self.findContours()

        outer = dmsh.Polygon(self.contour_points[0])
        inner = dmsh.Polygon(self.contour_points[3])
        geo = dmsh.Difference(outer,inner)

        X, cells, edges = dmsh.generate(geo, edge_size=self.edge_size)


        #try to further optimize the mesh
        try:
            #optimize it with centroidal voronoi tesselation
            self.mesh = dt.quasi_newton_uniform_full(X, cells, 1.0e-10, 100)

            # save as image

            self.mesh.save(
                "shape.png", show_coedges=False, show_axes=False, nondelaunay_edge_color="k",
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
    remove small contour details
    """


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


    def test_shape(self):
        im = cv.imread(self.img_file)
        edges = cv.Canny(im, 100,200)

        self.img_file = "edges_" + os.path.basename(self.img_file)
        self.display_img_file =  self.img_file
        cv.imwrite(self.display_img_file, edges)

        self.imgArea.update()

    def load_pic(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.display_img_file = self.img_file
        self.button1.setText(self.img_file)
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