import os
import sys
import subprocess

from PyQt5.QtWidgets import QWidget, QPushButton,\
    QApplication, QFileDialog, QGridLayout,  \
    QSlider, QGroupBox, QVBoxLayout, QLabel, QMainWindow, QScrollArea


from PyQt5.QtCore import Qt


import matplotlib.pyplot as plt
import numpy as np

from shapely.ops import polylabel

from image_view import ImageArea
from mesh_view import MeshArea
from knot_view import KnotArea
from segmentation import run_segmentation

"""
Main class inherited from a QMainWindow which is the main window of the program.
"""
class Window(QMainWindow):


    #display
    #TODO change to toggle
    knot_width = 3
    vertex_width = 10
    display_knot = True
    display_graph = True
    display_contour_points = False
    display_control_points = False
    selected_segment = 0

    #data structures
    triangle_mesh = {}
    quad_mesh = {}
    img_file = None
    display_img_file = None
    contour_points = []
    segments = []
    processed_img = None

    #parameters
    parameters = {
        "contour_roughness" : 0.05,
        "blur" : 2,
        "canny_th" : 0.2,
        "min_edge_size" : 15,
        "alpha" : 0.5
    }



    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 1000)
        self.init_ui()
        self.setup_graph()


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
        self.button1.clicked.connect(self.load_image)
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

        # button 8
        #TODO toggle button
        self.button8 = QPushButton("to quadrilaterals")
        self.button8.setCheckable(True)
        self.button8.clicked.connect(self.switch2quad)
        grid.addWidget(self.button8, 3, 1)

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

        # slider alpha
        self.slider6 = QSlider(Qt.Horizontal)
        self.slider6.setRange(0, 10)
        self.slider6.setSingleStep(1)
        self.slider6.setValue(self.alpha*10)
        # slider value label
        self.label6 = QLabel()
        self.slider6.valueChanged.connect(self.set_alpha)
        self.set_alpha()
        grid.addWidget(self.slider6, 5, 2)
        grid.addWidget(self.label6, 5, 3)

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

    def set_alpha(self):
        self.alpha = self.slider6.value()
        self.label6.setText("alpha " + str(self.alpha / 10))

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

    def run_segmentation(self):
        if self.img_file is None:
            self.load_image()

        self.segments, self.thin_segments, self.invalid, edges = run_segmentation(
            self.img_file, self.parameters["canny_th"], self.parameters["blur"],
            self.parameters["min_diameter"], self.parameters["min_arclength"]
        )

        # set slider range for amount of segments
        self.slider1.setRange(0, len(self.segments) - 1)

        #save image
        self.processed_img = edges

        #draw segments
        self.draw_segments()

    def run_meshing(self):
        if len(self.segments) == 0:
            self.run_segmentation()

    def switch2quad(self):
        self.setup_graph()
        self.triangle2quad()
        self.mesh2graph()
        self.meshArea.update()

    def draw_segments(self):
        # grayscale image back into RGB to draw contours
        plt.imshow(np.array(self.processed_img))

        # draw all too thin segments
        for polygon in self.thin_segments:
            plt.plot(*polygon.exterior.xy, color="blue", linewidth=1)

        # draw all invalid contours
        for polygon in self.invalid_segments:
            plt.plot(*polygon.exterior.xy, color="yellow", linewidth=1)

        # draw segments
        if len(self.segments) > 0:
            for polygon in self.segments:
                # draw all approximated contours except the selected level on image as green lines
                if polygon != self.segments[self.selected_segment]:
                    plt.plot(*polygon.exterior.xy, color="green", linewidth=1)
                # draw approximated selected contour on image as red lines

            polygon = self.segments[self.selected_segment]
            plt.plot(*polygon.exterior.xy, color="red", linewidth=2)
            for int in polygon.interiors:
                plt.plot(*int.xy, color="red", linewidth=2)
            try:
                label = polylabel(polygon, tolerance=1)
                plt.plot(label.x, label.y, color="red", linewidth=5, marker="X")
            except Exception:
                print(polygon, self.selected_segment)

        # contour image file
        self.display_img_file = "cont_" + os.path.basename(self.img_file)
        plt.savefig(self.display_img_file)
        plt.close()
        self.imgArea.update()


    def setup_graph(self):
        self.graph = {}
        self.graph["vertices"] = []
        self.graph["edges"] = []
        self.graph["colors"] = []

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.display_img_file = self.img_file
        self.button1.setText(os.path.basename(self.img_file))
        self.contour_points = []
        self.triangle_mesh = {}
        self.segments = []
        self.processed_img = None
        self.imgArea.update()

    def grabcut(self):
        subprocess.run(['python3 grab_cut.py \"' + self.img_file + '\"'], shell=True)
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