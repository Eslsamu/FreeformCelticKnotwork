import numpy as np
import scipy.spatial

def remove_close_vertices(X, min_dist):
    #compute distances between all vertices
    distances = []
    for v1 in X:
        d_row = []
        for v2 in X:
            if np.array_equal(v1,v2):
                d = np.inf
            else:
                d = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
            d_row.append(d)
        distances.append(d_row)

    #remove vertices till minimum distance of vertices is less than min_dist
    remove = []
    while True:
        #if a pair of vertices is too close then remove one of them
        if np.min(distances) < min_dist:
            #pair with shortest distance
            argmin = np.argmin(distances)
            v1,v2 = np.unravel_index(argmin, np.array(distances).shape)

            #find out which vertex is closer to its next neighbor
            d_v2, d_v1_2 = np.sort(distances[v1])[0:2]
            d_v1, d_v2_2 = np.sort(distances[v2])[0:2]

            #remove the vertex that is the closest to its next neighbor
            if d_v1_2 < d_v2_2:
                r = v1
            else:
                r = v2
            remove.append(r)

            #ignore the distance towards this vertex by setting it to inf
            for r,row in enumerate(distances):
                distances[r][v1] = np.inf
            distances[v1] = [np.inf for _ in distances[v1]]
        else:
            break

    #remove these vertices
    remove = list(set(remove))
    print(len(X))
    X = np.delete(X,remove,axis = 0)
    print(len(remove), len(X))

    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(X)
    cells = tri.simplices.copy()

    return X, cells

def mesh2graph(mesh):
    graph = {"vertices": [], "edges": [], "colors": []}
    for k, m in mesh.items():
        # max index of current graph
        last_index = len(graph["vertices"])

        # add vertices
        graph["vertices"].extend(m.node_coords.tolist())

        # shift index before adding new edges
        edges = m.edges["nodes"].tolist()
        for i, edge in enumerate(edges):
            for j, node in enumerate(edge):
                edges[i][j] += last_index
        # add edges
        graph["edges"].extend(edges)

        # each vertex is associated with a color
        # TODO let it be associated with the image surface
    return graph


def edges_from_cells(cells):
    edges = []
    for cell in cells:
        #get all edges in a cell
        for i in range(len(cell)):
            edge = [cell[i],cell[(i+1) % len(cell)]]

            #store edge if it wasn't stored before
            contains = False
            for other in edges:
                if (edge == other) or (edge == list(reversed(other))):
                    contains = True
            if not contains:
                edges.append(edge)
    return np.array(edges)

#find adjacent nodes in a lattice
#adjacent nodes
def adjacent_nodes():
    pass

#compute angle between two vectors independent from their direction
def angle(v1,v2):
    # get unit vectors
    u_v1 = v1 / np.linalg.norm(v1)
    u_v2 = v2 / np.linalg.norm(v2)

    # get angle in radians between the vectors (clip because of rounding errors)
    return np.arccos(np.clip(np.dot(u_v1, u_v2), -1.0, 1.0))

""" 

"""
def merge_border_cells(mesh, contour_points, th=0.8):
    while True:
        # select border cells
        border_cells = []
        for i, cell in enumerate(mesh.cells["nodes"]):
            border_nodes_index = []
            border_nodes_coords = []
            for node in cell:
                # check which nodes of cell touch the border
                if mesh.node_coords[node] in contour_points:
                    border_nodes_index.append(node)
                    border_nodes_coords.append(mesh.node_coords[node])

            if len(border_nodes_index) > 0:
                # store as [cell, cell index, [border node index], [border node coords], cell_quality]
                border_cells.append([cell, i, border_nodes_index, border_nodes_coords, mesh.cell_quality[i]])
        border_cells = np.array(border_cells, dtype=object)

        # stop when the cell quality of all border cells is bigger than the treshold
        if np.min(border_cells[:, 4]) > th:
            break

        # find the cell with lowest quality
        min_qual_i = np.argmin(border_cells[:, 4])
        min_qual_cell = border_cells[min_qual_i]

        # find neighbor border cells
        neighbors = []
        for i in min_qual_cell[2]:
            for cell in border_cells:
                if i in cell[2] and cell not in neighbors:
                    neighbors.append(cell)

        neighbors = np.array(neighbors)

        # option 1
        # merge remove a border node with low quality and then remesh --> super slow

        # option 2
        # merge with the neighbor that creates the highest quality cell

        # option 3
        # merge with the one that creates the least distortion in shape (based on knotwork)

        # option 4
        # remove the border node that results in the lowest quality based on
        # for each cell it belongs to / cells it belongs to
        # then retriangulate
        # or even reoptimize
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
        return mesh

#shift a numpy array by n steps
def shift(xs, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), xs[:-n]))
    else:
        return np.concatenate((xs[-n:], np.full(-n, np.nan)))
