import numpy as np
from shapely.geometry import box, Polygon, Point
import cv2
import random

#TODO
#take boundary points as input to define the shape (later with edge detection and now hand crafted)
#pack polygon grid with given resolution into the boundary as initial graph
#move outside vertices to closest point on the boundaru
#compute

#TODO add grid rotation
#only works for polygons without holes at the moment
"""
cellsize: unit in pixels
"""
def initial_lattice(cellsize = 50,boundary = Polygon([(0,0),(500,0),(0,500)])):
    #find rectangular boundary
    left = min(list(boundary.exterior.coords), key = lambda v: v[0])[0]
    right = max(list(boundary.exterior.coords), key=lambda v: v[0])[0]
    down = min(list(boundary.exterior.coords), key=lambda v: v[1])[1]
    up = max(list(boundary.exterior.coords), key=lambda v: v[1])[1]


    #graph to construct
    graph = {"vertices" : [], "edges" : []}
    for rows in range(int(np.abs(down-up)/cellsize)+1):
        for cols in range(int(np.abs(left-right)/cellsize)+1):
            #construct grid cell
            cell = box(cols * cellsize + left, rows * cellsize + down, (cols + 1) * cellsize + left, (rows + 1) * cellsize + down)
            #print(cell)
            #if cell is inside given shape boundary add it to the graph
            if boundary.contains(cell):
                #print("isinside",cell)
                #save vertex indices to construct edges afterwards
                indices = []
                for vertex in list(cell.exterior.coords)[:-1]:
                    if vertex not in graph["vertices"]:
                        graph["vertices"].append(vertex)
                        indices.append(len(graph["vertices"])-1)
                    else:

                        indices.append(graph["vertices"].index(vertex))
                graph["edges"].append((indices[0],indices[1]))
                graph["edges"].append((indices[0], indices[3]))
                graph["edges"].append((indices[1], indices[2]))
                graph["edges"].append((indices[3], indices[2]))
    return graph
"""
draw random pixel points of the interior of an object  using its contours
"""
#TODO adapt with watershed algorithm for interactive object segmentation
def init_pixels(img_file, threshold=150,amount=20, randomize = False):
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(imgray.shape, np.uint8)
    cv2.drawContours(mask, contours, 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    if randomize:
        lattice = random.sample(list(pixelpoints),amount)
    else:
        lattice = pixelpoints[::int(len(pixelpoints)/20)]

    return lattice, mask

"""
same as above but prints to the commandline to communicate with go 
"""
#TODO adapt with watershed algorithm for interactive object segmentation
def init_pixels_go(img_file, threshold=150,amount=128, randomize = False):
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(imgray.shape, np.uint8)
    cv2.drawContours(mask, contours, 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    if randomize:
        lattice = random.sample(list(pixelpoints),amount)
    else:
        lattice = pixelpoints[::int(len(pixelpoints)/20)]
    print(lattice)


