from mercat import compute_curves
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage, QBrush, QColor, QScreen, QPolygon, QTransform
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget, QApplication
import numpy as np
import json

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
        self.black_pen = QPen(Qt.black, self.parent.knot_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.red_pen = QPen(Qt.red, self.parent.knot_width/2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.green_pen = QPen(Qt.green, self.parent.knot_width/2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.blue_pen = QPen(Qt.blue, self.parent.knot_width / 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.yellow_pen = QPen(Qt.yellow, self.parent.knot_width / 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        self.qp.begin(self)
        scale = self.parent.scale_slider.value() * 0.3

        # adjust widget size based on loaded image
        img = self.parent.imgArea.image
        if img is not None:
            self.setMinimumWidth(img.width() * scale * 3)
            self.setMinimumHeight(img.height() * scale * 3)

        self.qp.drawImage(self.rect(), self.background, self.background.rect())

        # draw the knot

        self.shift = [500,500]
        self.qp.setPen(self.black_pen)
        self.draw_graph(scale)
        self.draw_knot(scale)



        self.qp.end()

    def take_screenshot(self):
        self.grab().save("knot","jpg")

    def draw_graph(self, scale,display_cell_quality = True):
        if self.parent.meshtype_button.isChecked():
            self.mesh = self.parent.tri_quad_mesh
        else:
            self.mesh = self.parent.triangle_mesh

        if self.mesh is None:
            return

        for i, submesh in self.mesh.items():
            cells = submesh.cells["nodes"]
            nodes = submesh.node_coords + self.shift

            if self.parent.display_edge_button.isChecked():
                for coords in nodes:
                    point = QPoint(coords[0] * scale, coords[1] * scale)
                    self.qp.drawPoint(point)

                for cell in cells:
                    points = [QPoint(nodes[cell[i]][0]*scale, nodes[cell[i]][1]*scale) for i in range(len(cell))]

                    # draw the cell
                    path = QPainterPath()
                    path.moveTo(points[0])
                    for j in range(1,len(points)):
                        path.lineTo(points[j])
                    path.closeSubpath()

                    # fill the cell with color intensity based on the cell quality
                    qual = submesh.cell_quality[i]
                    if display_cell_quality:
                        self.qp.setBrush(QBrush(QColor(0, 0, 255, int(qual * 255))))
                    else:
                        self.qp.setBrush(QBrush(QColor(0, 255, 0, 0)))
                    self.qp.drawPath(path)

    def draw_knot(self, scale):
        self.qp.setBrush(QBrush(QColor(0, 0, 0, 0)))
        if self.parent.meshtype_button.isChecked():
            mesh = self.parent.tri_quad_mesh
        else:
            mesh = self.parent.triangle_mesh

        if mesh is None:
            return

        colors = [self.red_pen,self.green_pen,self.blue_pen,self.yellow_pen]

        for k, submesh in mesh.items():
            """
            nodes = [[100,100],[200,100],[100,200],[200,200]]
            edges = [[0,1],[1,3],[0,2],[2,3]]
            curves = np.array(compute_curves(vertices=nodes,
                                             edges=edges,
                                             squish=self.parent.parameters["squish"]
                                             ))
            """
            curves = compute_curves(vertices = submesh.node_coords.tolist(),
                                    edges = submesh.edges["nodes"].tolist(),
                                    squish = self.parent.parameters["squish"]
                                    )

            over_under = {}
            count = 0
            for t, thread in curves.items():
                over = True
                path = QPainterPath()
                for i in range(len(thread)):
                    count += 1
                    if count > self.parent.knot_progression_slider.value():
                        pass

                    p0, p1, p2, p3 = np.array(thread[i][:-1]) * scale + self.shift
                    dir = thread[i][-1]

                    path.moveTo(p0[0],p0[1])
                    path.cubicTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

                    #mark each junction
                    key = json.dumps(p0.tolist())

                    if not key in over_under.keys():
                        over_under[key] = (p0, p1, dir,t)
                    else:
                        #if junction was crossed before then create overlap
                        #draw two line next to each other at the junction with
                        #direction of the current curve
                        if over:
                            over_under[key] = (p0,p1, dir,t)
                            over = False
                        else:
                            over = True

                    if self.parent.control_points_button.isChecked():
                        self.qp.setPen(self.red_pen)

                        self.qp.drawLine(p0[0],p0[1],p1[0],p1[1])
                        self.qp.drawLine(p3[0], p3[1], p2[0], p2[1])

                        self.qp.setPen(self.yellow_pen)
                        self.qp.drawPoint(p1[0], p1[1])
                        self.qp.drawPoint(p2[0], p2[1])

                        self.qp.setPen(self.black_pen)
                        self.qp.drawPoint(p0[0], p0[1])
                        self.qp.drawPoint(p3[0], p3[1])

                self.qp.setPen(self.black_pen)
                self.qp.drawPath(path)
                self.qp.setPen(colors[t%len(colors)])
                self.qp.drawPath(path)

            for k, crossing in over_under.items():
                if crossing:
                    p0, p1, dir, t = crossing
                    p0 = np.array(p0)
                    p1 = np.array(p1)
                    uv = (p1 - p0) / np.linalg.norm([p1-p0])

                    #counter clockwise rotation
                    uv_rotated = np.array([-uv[1],
                                  uv[0]])

                    a = p0 + uv * self.parent.knot_width/3 + uv_rotated * self.parent.knot_width/3
                    b = p0 - uv * self.parent.knot_width/3 + uv_rotated * self.parent.knot_width/3
                    c = p0 + uv * self.parent.knot_width/3 - uv_rotated * self.parent.knot_width/3
                    d = p0 - uv * self.parent.knot_width/3 - uv_rotated * self.parent.knot_width/3
                    try:
                        self.qp.setPen(colors[t % len(colors)])
                        rect = QPolygon(
                            [QPoint(int(a[0]), int(a[1])), QPoint(int(b[0]), int(b[1])), QPoint(int(c[0]), int(c[1])),
                                                                                                    QPoint(int(d[0]), int(d[1]))])
                        self.qp.drawPolygon(rect)
                        self.qp.setPen(QPen(Qt.black, self.parent.knot_width/3.5, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                        self.qp.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
                        self.qp.drawLine(int(c[0]),int(c[1]),int(d[0]),int(d[1]))
                    except ValueError:
                        continue #TODO fix NaN






