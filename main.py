import os
import sys
import subprocess

from PyQt5.QtWidgets import QWidget, QPushButton,\
    QApplication, QFileDialog, QGridLayout,  \
    QSlider, QGroupBox, QLabel, QMainWindow, QScrollArea, QComboBox


from PyQt5.QtCore import Qt


import matplotlib.pyplot as plt
import numpy as np


from image_view import ImageArea
from mesh_view import MeshArea
from knot_view import KnotArea
from segmentation import run_segmentation
from meshing import mesh_segments
from quad_conversion import triangle_to_3_4_greedy, triangle_to_3_4_browne


"""
Main class inherited from a QMainWindow which is the main window of the program.
"""
class Window(QMainWindow):


    #display
    #TODO change to toggle
    knot_width = 10
    vertex_width = 10
    display_contour_points = False
    display_control_points = False
    selected_segment = 0

    #data structures
    triangle_mesh = None
    tri_quad_mesh = None
    img_file = "test_images/shape/circle.png"
    display_img_file = "test_images/shape/circle.png"
    contour_points = []
    segments = []
    processed_img = None

    #parameters
    parameters = {
        "contour_roughness" : 100,
        "blur" : 1,
        "squish" : 0.5,
        "canny_th" : 0.2,
        "min_edge_size" : 100,
        "alpha" : 0.5,
        "min_diameter" : 0,
        "min_arclength" : 0,
        "join_th" : 0.6
    }



    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 1000)
        self.init_ui()



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
        self.button1 = QPushButton("load image")
        self.button1.clicked.connect(self.load_image)
        grid.addWidget(self.button1, 0, 0)

        # button 3
        self.button3 = QPushButton("run meshing")
        self.button3.clicked.connect(self.run_meshing)
        grid.addWidget(self.button3, 2,0)

        #button 5
        self.button5 = QPushButton("mask image")
        self.button5.clicked.connect(self.grabcut)
        grid.addWidget(self.button5, 0, 1)

        #button 6
        self.button6 = QPushButton("run segmentation")
        self.button6.clicked.connect(self.run_segmentation)
        grid.addWidget(self.button6, 1,1)

        # button 7
        self.button7 = QPushButton("show contour points")
        self.button7.clicked.connect(self.show_contour_points)
        grid.addWidget(self.button7,2,1)

        # button 8
        self.button8 = QPushButton("run quad conversion")
        self.button8.clicked.connect(self.run_quad_conversion)
        grid.addWidget(self.button8, 3, 0)

        # button 9
        self.meshtype_button = QPushButton("show quad mesh")
        self.meshtype_button.setCheckable(True)
        self.meshtype_button.clicked.connect(lambda : self.update())
        grid.addWidget(self.meshtype_button, 1, 0)

        # button 10
        self.control_points_button = QPushButton("show control points")
        self.control_points_button.setCheckable(True)
        self.control_points_button.clicked.connect(lambda: self.update())
        grid.addWidget(self.control_points_button, 3, 1)

        # slider contour level
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 10)
        self.slider1.setValue(self.selected_segment)
        # slider value label
        self.label1 = QLabel()
        self.slider1.valueChanged.connect(self.set_selected_segment)
        grid.addWidget(self.slider1,0,2)
        grid.addWidget(self.label1,0,3)

        # slider edge size
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(5, 200)
        self.slider2.setValue(self.parameters["min_edge_size"])
        # slider value label
        self.label2 = QLabel()
        self.slider2.valueChanged.connect(self.set_params)
        grid.addWidget(self.slider2,1,2)
        grid.addWidget(self.label2,1,3)

        # slider knot width
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(1, 30)
        self.slider3.setValue(self.knot_width)
        # slider value label
        self.label3 = QLabel()
        self.slider3.valueChanged.connect(self.set_knot_width)
        grid.addWidget(self.slider3,2,2)
        grid.addWidget(self.label3,2,3)

        # slider contour roughness
        self.contour_roughness_slider = QSlider(Qt.Horizontal)
        self.contour_roughness_slider.setRange(0, 2000)
        self.contour_roughness_slider.setSingleStep(1)
        self.contour_roughness_slider.setValue(self.parameters["contour_roughness"])
        # slider value label
        self.contour_roughness_label = QLabel()
        self.contour_roughness_slider.valueChanged.connect(self.set_params)
        grid.addWidget(self.contour_roughness_slider,3,2)
        grid.addWidget(self.contour_roughness_label,3,3)

        # slider blur
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setRange(0, 10)
        self.slider5.setSingleStep(1)
        self.slider5.setValue(self.parameters["blur"])
        # slider value label
        self.label5 = QLabel()
        self.slider5.valueChanged.connect(self.set_params)
        grid.addWidget(self.slider5, 4, 2)
        grid.addWidget(self.label5, 4, 3)

        # slider alpha
        self.slider6 = QSlider(Qt.Horizontal)
        self.slider6.setRange(0, 10)
        self.slider6.setSingleStep(1)
        self.slider6.setValue(self.parameters["alpha"]*10)
        # slider value label
        self.label6 = QLabel()
        self.slider6.valueChanged.connect(self.set_params)
        grid.addWidget(self.slider6, 0, 4)
        grid.addWidget(self.label6, 0, 5)

        # slider squish
        self.slider7 = QSlider(Qt.Horizontal)
        self.slider7.setRange(1, 10)
        self.slider7.setSingleStep(1)
        self.slider7.setValue(self.parameters["squish"] * 10)
        # slider value label
        self.label7 = QLabel()
        self.slider7.valueChanged.connect(self.set_squish)
        grid.addWidget(self.slider7, 1, 4)
        grid.addWidget(self.label7, 1, 5)

        #combobox generator
        self.generator_combobox = QComboBox()
        self.generator_combobox.addItem("wavefront")
        self.generator_combobox.addItem("dmsh")
        grid.addWidget(self.generator_combobox, 4, 4)

        # button 11
        self.cvt_button = QPushButton("cvt smoothing")
        self.cvt_button.setCheckable(True)
        grid.addWidget(self.cvt_button, 3, 4)

        # button 12
        self.display_edge_button = QPushButton("display edges")
        self.display_edge_button.setCheckable(True)
        grid.addWidget(self.display_edge_button, 4, 5)

        # button 13
        self.knot_screenshot_button = QPushButton("save knot")
        self.knot_screenshot_button.clicked.connect(self.knotArea.take_screenshot)
        grid.addWidget(self.knot_screenshot_button, 3, 5)

        # slider scale
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 10)
        self.scale_slider.setSingleStep(1)
        self.scale_slider.setValue(1)
        # slider value label
        self.scale_label = QLabel()
        self.scale_label.setText("knot scale 1")
        self.scale_slider.valueChanged.connect(
            lambda : self.scale_label.setText("knot scale "+str(self.scale_slider.value())))
        grid.addWidget(self.scale_slider, 2, 4)
        grid.addWidget(self.scale_label, 2, 5)

        # slider knot progression
        self.knot_progression_slider = QSlider(Qt.Horizontal)
        self.knot_progression_slider.setRange(0, 500)
        self.knot_progression_slider.setSingleStep(1)
        self.knot_progression_slider.setValue(500)
        # slider value label
        self.knot_progression_label = QLabel()
        self.knot_progression_label.setText("knot progression")
        self.knot_progression_slider.valueChanged.connect(
            lambda: self.knot_progression_label.setText("knot progression " + str(self.knot_progression_slider.value())))
        grid.addWidget(self.knot_progression_slider, 4, 0)
        grid.addWidget(self.knot_progression_label, 4, 1)

        # combobox converter
        self.converter_combobox = QComboBox()
        self.converter_combobox.addItem("greedy")
        self.converter_combobox.addItem("browne")
        grid.addWidget(self.converter_combobox, 0, 6)

        # slider alpha
        self.join_th_slider = QSlider(Qt.Horizontal)
        self.join_th_slider.setRange(0, 10)
        self.join_th_slider.setSingleStep(1)
        self.join_th_slider.setValue(self.parameters["join_th"] * 10)
        # slider value label
        self.join_th_label = QLabel()
        self.join_th_slider.valueChanged.connect(self.set_params)
        grid.addWidget(self.join_th_slider, 1, 6)
        grid.addWidget(self.join_th_label, 1, 7)

        self.set_knot_width()
        self.set_selected_segment()
        self.set_squish()
        self.set_params()

        return groupbox

    def paintEvent(self, event):
        self.meshArea.update()
        self.imgArea.update()
        self.knotArea.update()


    def set_selected_segment(self):
        self.selected_segment = self.slider1.value()
        self.label1.setText("display segment " + str(self.selected_segment))

        if self.processed_img is not None:
            self.draw_segments()

        self.imgArea.update()

    def set_knot_width(self):
        self.knot_width = self.slider3.value()
        self.label3.setText("knot width " + str(self.knot_width))
        self.update()

    def set_squish(self):
        self.parameters["squish"] = self.slider7.value() / 10
        self.label7.setText("curve squish " + str(self.parameters["squish"]))
        self.update()

    def set_params(self):
        self.parameters["min_edge_size"] = self.slider2.value()
        self.label2.setText("min_edge_size " + str(self.parameters["min_edge_size"]))

        self.parameters["contour_roughness"] = self.contour_roughness_slider.value()
        self.contour_roughness_label.setText("contour roughness " + str(self.parameters["contour_roughness"]))

        self.parameters["blur"] = self.slider5.value()
        self.label5.setText("blur " + str(self.parameters["blur"]))

        self.parameters["alpha"] = self.slider6.value() / 10
        self.label6.setText("direction vs regularity " + str(self.parameters["alpha"]))

        self.parameters["join_th"] = self.join_th_slider.value() / 10
        self.join_th_label.setText("join triangle th" + str(self.parameters["join_th"]))

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

        self.segments, self.thin_segments, self.invalid_segments, edges = run_segmentation(
            self.img_file, self.parameters["canny_th"], self.parameters["blur"],
            min_diameter = self.parameters["min_edge_size"]/2,
            min_arclength = self.parameters["min_edge_size"] * 3,
            contour_roughness = self.parameters["contour_roughness"]
        )

        # set slider range for amount of segments
        self.slider1.setRange(0, len(self.segments)-1)

        #save image
        self.processed_img = edges

        #draw segments
        self.draw_segments()


    def run_meshing(self):
        if len(self.segments) == 0:
            self.run_segmentation()
        self.triangle_mesh = mesh_segments(self.segments,
                                           self.parameters["min_edge_size"],
                                           generator = self.generator_combobox.currentText(),
                                           cvt = self.cvt_button.isChecked())
        self.knot_progression_slider.setRange(0,int(len(self.triangle_mesh.get(0).node_coords)*5))
        self.meshArea.update()

    def run_quad_conversion(self):
        print("convert to quads")
        #self.tri_quad_mesh = triangle2quad(self.triangle_mesh, self.segments, self.parameters["alpha"])
        if self.converter_combobox.currentText() == "greedy":
            self.tri_quad_mesh = triangle_to_3_4_greedy(self.triangle_mesh,
                                                        segments = self.segments, alpha=self.parameters["alpha"])
        if self.converter_combobox.currentText() == "browne":
            self.tri_quad_mesh = triangle_to_3_4_browne(self.triangle_mesh, th=self.parameters["join_th"])
        self.meshtype_button.setChecked(True)
        print("finished conversion")
        self.update()

    def draw_segments(self):
        # grayscale image back into RGB to draw contours
        plt.imshow(np.array(self.processed_img))

        # draw all too thin segments
        for segment in self.thin_segments:
            polygon = segment["polygon"]
            plt.plot(*polygon.exterior.xy, color="blue", linewidth=1)

        # draw all invalid contours
        for polygon in self.invalid_segments:
            plt.plot(*polygon.exterior.xy, color="yellow", linewidth=1)

        # draw segments
        if len(self.segments) > 0:
            for segment in self.segments:
                polygon = segment["polygon"]

                # draw all approximated contours except the selected level on image as green lines
                if polygon != self.segments[self.selected_segment]:
                    plt.plot(*polygon.exterior.xy, color="green", linewidth=1)
                # draw approximated selected contour on image as red lines

            #draw selected segment in red
            polygon = self.segments[self.selected_segment]["polygon"]
            plt.plot(*polygon.exterior.xy, color="red", linewidth=2)
            for int in polygon.interiors:
                plt.plot(*int.xy, color="red", linewidth=2)
            try:
                label = self.segments[self.selected_segment]["polylabel"]
                plt.plot(label.x, label.y, color="red", linewidth=5, marker="C")
            except Exception:
                print(polygon, self.selected_segment)

        # contour image file
        self.display_img_file = "cont_" + os.path.basename(self.img_file)
        plt.savefig(self.display_img_file)
        plt.close()
        self.imgArea.update()

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.img_file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.display_img_file = self.img_file
        self.button1.setText(os.path.basename(self.img_file))
        self.triangle_mesh = None
        self.tri_quad_mesh = None
        self.processed_img = None
        self.segments = []
        self.imgArea.update()
        self.meshArea.update()
        self.knotArea.update()

    def grabcut(self):
        subprocess.run(['python3 grab_cut.py \"' + self.img_file + '\"'], shell=True)
        self.img_file = "mask_" + os.path.basename(self.img_file)
        self.display_img_file = self.img_file
        self.imgArea.update()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    import faulthandler
    faulthandler.enable()
    sys.exit(app.exec_())
    window.run_meshing()

if __name__ == '__main__':
    main()