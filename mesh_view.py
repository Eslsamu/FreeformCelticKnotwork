from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage, QColor, QBrush
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget


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
        self.qp.setPen(self.black_pen)
        self.draw_graph()


        #mark the contour points
        if self.parent.display_contour_points:
            for x, y in self.parent.boundary_points[self.parent.selected_segment]:
                self.qp.setPen(self.green_pen)
                self.qp.drawPoint(x,y)

        #adjust widget size based on loaded image
        img = self.parent.imgArea.image
        if img is not None:
            self.setMinimumWidth(img.width()*1.2)
            self.setMinimumHeight(img.height() * 1.2)

        self.qp.end()


    def draw_graph(self, display_cell_quality = True, display_edges = False):
        if self.parent.meshtype_button.isChecked():
            mesh = self.parent.quad_mesh
        else:
            mesh = self.parent.triangle_mesh

        if mesh is None:
            return

        for i, submesh in mesh.items():
            cells = submesh.cells["nodes"]
            nodes = submesh.node_coords

            if display_edges is False:
                for coords in nodes:
                    point = QPoint(coords[0],coords[1])
                    self.qp.drawPoint(point)
            else:

                for cell in cells:
                    points = [QPoint(coords[cell[i]][0], coords[cell[i]][1]) for i in range(len(cell))]

                    # draw the cell
                    path = QPainterPath()
                    path.moveTo(points[0])
                    for j in range(1,len(points)):
                        path.lineTo(points[j])
                    path.closeSubpath()

                    # fill the cell with color intensity based on the cell quality
                    qual = submesh.cell_quality[i]
                    if display_cell_quality:
                        self.qp.setBrush(QBrush(QColor(0, 255, 0, int(qual * 255))))
                    else:
                        self.qp.setBrush(QBrush(QColor(0, 255, 0, 0)))
                    self.qp.drawPath(path)

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