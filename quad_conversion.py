import numpy as np
from helpers import edges_from_cells,  angle
from shapely.ops import nearest_points
from shapely.geometry import Point

# paper: Automatic Conversion of Triangular Meshes Into Quadrilateral Meshes with Directionality
# (slightly adapted)
def triangle_to_3_4_greedy(triangle_mesh, segments, alpha):
    tri_quad_mesh = {}
    for k, submesh in triangle_mesh.items():
        cells = submesh.cells["nodes"]
        node_coords = submesh.node_coords

        # find neighbor cells of each cell
        neighbors = {}
        for triangle in cells:
            for other_triangle in cells:
                if np.any(triangle != other_triangle):
                    shared_nodes = 0
                    for node in triangle:
                        if node in other_triangle:
                            shared_nodes += 1
                    if shared_nodes == 2:
                        try:
                            neighbors[str(triangle)].append(other_triangle)
                        except KeyError:
                            neighbors[str(triangle)] = [other_triangle]

        # find all possible quads and compute their quality
        possible_quads = []
        for triangle in cells:

            try:
                # check if cell has any neighbors
                for neighbor in neighbors[str(triangle)]:
                    # form new quad
                    dist_node1 = [node for node in neighbor if node not in triangle][0]
                    dist_node2 = [node for node in triangle if node not in neighbor][0]
                    shrd_nodes = [node for node in triangle if node in neighbor]
                    # order the nodes correctly
                    nodes = [dist_node1, shrd_nodes[0], dist_node2, shrd_nodes[1]]

                    # check if this quad was already added
                    if not any(all(node in quad["nodes"] for node in nodes) for quad in possible_quads):
                        coords = [node_coords[node] for node in nodes]
                        qual = alpha * geometric_irregularity(coords) + \
                               (1 - alpha) * directionality_error(coords, segments[k]["polygon"])
                        possible_quads.append({"nodes": nodes, "coords": coords,
                                               "qual": qual, "triangles": [triangle, neighbor]})
            except KeyError: #skip if triangle has no neighbors (will remove cell from resulting mesh)
                continue

        # sort possible quads by their quality
        possible_quads = sorted(possible_quads, key=lambda quad: quad["qual"])

        # keep the best quads
        chosen_quads = []
        taken_triangles = []
        for quad in possible_quads:
            triangle1 = quad["triangles"][0]
            triangle2 = quad["triangles"][1]

            if len(chosen_quads) > 0:
                # check if any triangle of this is already taken by a better quad
                if not next((True for elem in taken_triangles if np.array_equal(elem, triangle1)), False) \
                        and not next((True for elem in taken_triangles if np.array_equal(elem, triangle2)), False):
                    chosen_quads.append(quad)
                    taken_triangles.append(triangle1)
                    taken_triangles.append(triangle2)
            else:
                chosen_quads.append(quad)

        # triangles that are left over and can't be converted to quads
        remaining_triangles = []
        for triangle in cells:
            if not next((True for elem in taken_triangles if np.array_equal(elem, triangle)), False):
                coords = [node_coords[node] for node in triangle]
                taken_triangles.append(triangle)
                remaining_triangles.append(
                    {"nodes": triangle, "coords": coords, "qual": 0, "triangles": [triangle]})

        #store cells
        cells = [quad["nodes"] for quad in chosen_quads]
        cells.extend([triangle["nodes"] for triangle in remaining_triangles])
        #store cell quality
        qual = [quad["qual"] for quad in chosen_quads]
        qual.extend([triangle["qual"] for triangle in remaining_triangles])

        #compute edges
        edges = edges_from_cells(cells)

        #store submesh
        tri_quad_mesh[k] = Tri_Quad_Mesh(node_coords,
                                cells= cells,
                                cell_quality = qual,
                                edges = edges
                                )
    return tri_quad_mesh

#standard deviation of egde lengths
def geometric_irregularity(nodes, mode = "min-angle"):

    if mode == "std-edges":
        dist = [np.sqrt((nodes[i-1][0] - nodes[i][0]) ** 2 + (nodes[i-1][1] - nodes[i][1]) ** 2)
                for i in range(len(nodes))]
        result = np.std(dist) / np.mean(dist)
    if mode == "min-angle":
        angles = []
        for i in range(len(nodes)):
            # edge as vector
            v1 = nodes[i - 1] - nodes[i]
            v2 = nodes[i] - nodes[(i + 1) % len(nodes)]
            angles.append(angle(v1,v2))
        result = min(angles) / np.pi
    return result



#edge should ideally be normal or tangent to its closest boundary point
def directionality_error(nodes, polygon):
    angles = []
    for i in range(len(nodes)):
        #edge as vector
        v1 = nodes[i-1] - nodes[i]

        #get vector of two closest boundary points to each node
        v2 = nearest_points(polygon,Point(nodes[i][0], nodes[i][1]))[0] -\
             nearest_points(polygon,Point(nodes[i-1][0], nodes[i-1][1]))[0]

        angles.append(angle(v1,v2) % (0.5 * np.pi))

    return np.mean(angles) / ( 0.5 * np.pi)

#automatic font decoration
def triangle_to_3_4_browne(triangle_mesh, th=0.7):
    tri_quad_mesh = {}
    for k, submesh in triangle_mesh.items():
        cells = submesh.cells["nodes"]
        node_coords = submesh.node_coords

        # find neighbor cells of each cell
        neighbors = {}
        for triangle in cells:
            for other_triangle in cells:
                if np.any(triangle != other_triangle):
                    shared_nodes = 0
                    for node in triangle:
                        if node in other_triangle:
                            shared_nodes += 1
                    if shared_nodes == 2:
                        try:
                            neighbors[str(triangle)].append(other_triangle)
                        except KeyError:
                            neighbors[str(triangle)] = [other_triangle]

        taken_triangles = []

        # find all optimal quads
        optimal_quads = []
        for triangle in cells:
            try:
                # check if cell has any neighbors
                for neighbor in neighbors[str(triangle)]:
                    if not next((True for elem in taken_triangles if np.array_equal(elem, triangle)), False) \
                            and not next((True for elem in taken_triangles if np.array_equal(elem, neighbor)), False):
                        # form new quad
                        dist_node1 = [node for node in neighbor if node not in triangle][0]
                        dist_node2 = [node for node in triangle if node not in neighbor][0]
                        shared_nodes = [node for node in triangle if node in neighbor]
                        # order the nodes correctly
                        nodes = [dist_node1, shared_nodes[0], dist_node2, shared_nodes[1]]
                        coords = [node_coords[node] for node in nodes]
                        mid_vec = node_coords[shared_nodes[0]] - node_coords[shared_nodes[1]]
                        if is_optimal_quad(coords,mid_vec):
                            optimal_quads.append({"nodes": nodes, "coords": coords,
                              "qual": 1, "triangles": [triangle, neighbor]})
                            taken_triangles.append(triangle)
                            taken_triangles.append(neighbor)


            except KeyError:  # skip if triangle has no neighbors (will remove cell from resulting mesh)
                #TODO keep it as single triangle
                continue

        # find all good quads
        good_quads = []
        for triangle in cells:
            try:
                # check if cell has any neighbors
                for neighbor in neighbors[str(triangle)]:

                    if not next((True for elem in taken_triangles if np.array_equal(elem, triangle)), False) \
                            and not next((True for elem in taken_triangles if np.array_equal(elem, neighbor)),
                                         False):
                        # form new quad
                        dist_node1 = [node for node in neighbor if node not in triangle][0]
                        dist_node2 = [node for node in triangle if node not in neighbor][0]
                        shrd_nodes = [node for node in triangle if node in neighbor]
                        # order the nodes correctly
                        nodes = [dist_node1, shrd_nodes[0], dist_node2, shrd_nodes[1]]
                        coords = [node_coords[node] for node in nodes]
                        qual = quad_qual(coords)
                        if qual > th:
                            good_quads.append({"nodes": nodes, "coords": coords,
                                                           "qual": qual, "triangles": [triangle, neighbor]})
                            taken_triangles.append(triangle)
                            taken_triangles.append(neighbor)

            except KeyError:  # skip if triangle has no neighbors (will remove cell from resulting mesh)
                # TODO keep it as single triangle
                continue

        #triangles that are left over and can't be converted to quads
        remaining_triangles = []
        for triangle in cells:
            if not next((True for elem in taken_triangles if np.array_equal(elem, triangle)), False):
                coords = [node_coords[node] for node in triangle]
                taken_triangles.append(triangle)
                remaining_triangles.append({"nodes": triangle,"coords": coords, "qual":0, "triangles": [triangle]})

        # store cells
        cells = [quad["nodes"] for quad in optimal_quads]
        cells.extend([quad["nodes"] for quad in good_quads])
        cells.extend([triangle["nodes"] for triangle in remaining_triangles])

        # store cell quality
        qual = [quad["qual"] for quad in optimal_quads]
        qual.extend([quad["qual"] for quad in good_quads])
        qual.extend([triangle["qual"] for triangle in remaining_triangles])

        # compute edges
        edges = edges_from_cells(cells)

        # store submesh
        tri_quad_mesh[k] = Tri_Quad_Mesh(node_coords,
                                         cells=cells,
                                         cell_quality=qual,
                                         edges=edges
                                         )
    return tri_quad_mesh


#ask if this makes sense, because it allows quads to be really skewed (or does delaunay discourage that?)
def is_optimal_quad(coords, mid_vec):
    vec = np.array([coords[i] - coords[i - 1] for i in range(len(coords))])
    length = np.array([np.linalg.norm([v]) for v in vec])
    mid_length = np.linalg.norm([mid_vec])
    if np.max(length) < mid_length:
        return True
    else:
        return False

#th should be ~ 0.7 to 0.9
def quad_qual(coords):
    vec = np.array([coords[i]-coords[i-1] for i in range(len(coords))])
    a = np.array([angle(vec[i],vec[(i-1)]) for i in range(len(vec))])
    length = np.array([np.linalg.norm([v]) for v in vec])

    qual = 1-np.sum([np.abs(a - np.pi/2)/(np.pi/2) + np.abs(length - np.mean(length))/np.mean(length)])/8

    return qual

class Tri_Quad_Mesh:
    def __init__(self, node_coords=None, cells=None, cell_quality=None, edges=None):
        self.cells = {"nodes": cells}
        self.node_coords = node_coords
        self.cell_quality = cell_quality
        self.edges = {"nodes": edges}

def main():
    """
   test

   cells = [np.array([0,1,3]),np.array([1,3,4]),np.array([1,4,2]),np.array([4,5,2]),
                np.array([3,4,6]),np.array([6,7,4]),np.array([4,5,7]),np.array([5,7,8])]
   node_coords = [np.array([100,100]),np.array([150,100]),np.array([200,100]),
                  np.array([100,150]),np.array([150,150]),np.array([200,150]),
                  np.array([100,200]),np.array([150,200]),np.array([200,200])]
    """
    from scipy.spatial import Delaunay
    from shapely.ops import triangulate
    from shapely.geometry import MultiPoint

    node_coords = [np.array([100,100]),np.array([150,100]),np.array([200,100]),
                  np.array([100,150]),np.array([150,150]),np.array([200,150]),
                  np.array([100,200]),np.array([150,200]),np.array([170,170])]
    cells = Delaunay(node_coords).simplices
    test = {0:Tri_Quad_Mesh(
        node_coords=node_coords,
        cells=cells
    )}
    q = triangle_to_3_4_browne(test,th=0.8)
    print(q.get(0).cells)

if __name__ == '__main__':
    main()








