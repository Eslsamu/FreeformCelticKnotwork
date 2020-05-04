import numpy as np
from shapely.geometry import box, Polygon, Point
from scipy.optimize import minimize
from skimage.io import imread

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

#TODO
def random_points():
    pass


