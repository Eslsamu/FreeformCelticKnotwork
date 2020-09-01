import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point
import dmsh
import matplotlib.pyplot as plt
import optimesh
import meshplex
from helpers import angle,edges_from_cells, adjacent_nodes, shift

"""
Mesh each segment and skip segments enclosed in other segments to display object features

returns a dictionary of triangles meshes for each segment
"""
def mesh_segments(segments, edge_size):
    triangle_mesh = {}
    # TODO mesh each segment with adaptive edge size
    skipped = []
    for i, segment in enumerate(segments):
        print("meshing segment", i, "/", len(segments))

        polygon = segment["polygon"]

        # skip segments a that are enclosed in another segment b which was not skipped before
        enclosing = None
        for j in range(i):
            b = segments[j]["polygon"]
            a = polygon
            b_ext = list(b.exterior.coords.xy)
            b_without_holes = Polygon([[b_ext[0][i], b_ext[1][i]] for i in range(len(b_ext[0]))])
            if b_without_holes.contains(a):
                enclosing = segments[j]
                print(j, "enclosing", i)
        if enclosing not in skipped and enclosing is not None:
            print("skip", i)
            skipped.append(segment)
            continue
        print(i, "didnt get skipped")



        # inspect segment
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        plt.plot(*polygon.exterior.xy)
        for int in polygon.interiors:
            plt.plot(*int.xy, color="red")
        plt.savefig(str(i) + "_shape.png")
        plt.close()

        geo = polygon2geo(polygon)

        # inspect geo
        geo.save(str(i) + "_geo.png")


        X, cells = dmsh.generate(geo, edge_size=edge_size, verbose = False)
        print("nodes", X.shape, "cells", cells.shape)

        #print("remove close vertices ...")
        # X, cells = self.remove_close_vertices(X,cells, self.min_edge_size/3)
        #print("nodes", X.shape, "cells", cells.shape)

        print("optimize further with cvt ...")

        # try to further optimize the mesh
        try:
            # optimize it with centroidal voronoi tesselation
            X, cells = optimesh.cvt.quasi_newton_uniform_full(X, cells, 1.0e-10, 100)
        except meshplex.exceptions.MeshplexError as err:
            print(err)

        mesh = meshplex.MeshTri(X, cells)

        # save as image
        mesh.save(
            str(i) + "_mesh.png", show_coedges=False, show_axes=False,
        )

        # save mesh
        triangle_mesh[i] = mesh
    print("finished meshing")
    return triangle_mesh

def polygon2geo(polygon):
    # exterior boundary (dmsh does not include the last point twice to close the polygon such as shapely)
    e = list(polygon.exterior.coords.xy)
    boundary_e = [[e[0][i], e[1][i]] for i in range(len(e[0]) - 1)]
    outer = dmsh.Polygon(boundary_e)

    # interior boundary
    inner = []
    for int in polygon.interiors:
        h = list(int.coords.xy)
        boundary_h = [[h[0][i], h[1][i]] for i in range(len(h[0]) - 1)]
        hole = dmsh.Polygon(boundary_h)
        inner.append(hole)

    if len(inner) > 1:
        inner = dmsh.Union(inner)
        geo = dmsh.Difference(outer, inner)
    elif len(inner) == 1:
        inner = inner[0]
        geo = dmsh.Difference(outer, inner)
    else:
        geo = outer
    return geo

class QuadMesh:
    def __init__(self, node_coords=None, cells=None, cell_quality=None, edges=None):
        self.cells = {"nodes": cells}
        self.node_coords = node_coords
        self.cell_quality = cell_quality
        self.edges = {"nodes": edges}

#wavefront meshing
def wavefront_step(seed,stepsize):
    mesh = QuadMesh(node_coords= seed)
    nodes = mesh.node_coords

    x = nodes[:, 0]
    y = nodes[:, 1]

    vec_x = np.diff(x,prepend=x[-1])
    rev_vec_y = np.diff(y,prepend=y[-1]) * -1

    #(x2 +- y1-y2 * (n/m), y2 +- x2-x1 * (n/m)) --> xy3 in right angled, regular triangle
    inside = -1
    distances = np.linalg.norm([vec_x,rev_vec_y],axis=0)
    # handle div by zero
    scale = np.divide(stepsize,distances, out= np.zeros_like(distances), where=distances != 0)
    new_x = x + rev_vec_y * inside * scale
    new_y = y + vec_x * inside * scale
    return np.column_stack([new_x,new_y])

def main():
    seed = np.array([[1,1],[1.5,1.5],[2,2],[2,1]])
    print(wavefront_meshing(seed))

if __name__ == '__main__':
    main()




#triangle mesh lattice into quadrilateral mesh
def lattice2quad(triangle_mesh):
    pass

#wavefront quadrilateral meshing
def wavefront_meshing(segments):
    pass

#quadrileteral mesh smoothing
#TODO first
def smooth_quads(quad_mesh, eps):
    mesh = QuadMesh(np.array()) #example mesh for testing
    old_pos = None

    while True:
        if (old_pos - mesh.node_coords.copy()) < eps:
            break
        """
        #compute desired length
        edges_vec = mesh.node_coords[edges[:, 1]] - mesh.node_coords[edges[:, 0]]
        edge_lengths = np.sqrt(numpy.einsum("ij,ij->i", edges_vec, edges_vec))
        edges_vec /= edge_lengths[..., None]

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (
                                 mesh.node_coords[edges[:, 1]] + mesh.node_coords[edges[:, 0]]
                         ) / 2
        p = edge_size_function(edge_midpoints.T)
        desired_lengths = (
                f_scale
                * p
                * numpy.sqrt(numpy.dot(edge_lengths, edge_lengths) / numpy.dot(p, p))
        )

        force_abs = desired_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = edges_vec * force_abs[..., None]

        # bincount replacement for the slow numpy.add.at
        # more speed-up can be achieved if the weights were contiguous in memory, i.e.,
        # if force[k] was used
        n = mesh.node_coords.shape[0]
        force_per_node = numpy.array(
            [
                numpy.bincount(edges[:, 0], weights=-force[:, k], minlength=n)
                + numpy.bincount(edges[:, 1], weights=+force[:, k], minlength=n)
                for k in range(force.shape[1])
            ]
        ).T

        update = delta_t * force_per_node
        """

#TODO second
def cvt_quad(quadmesh):
    pass

# paper: Automatic Conversion of Triangular Meshes Into Quadrilateral Meshes with Directionality
# (slightly adapted)
def triangle2quad(triangle_mesh, segments, alpha):

    quad_mesh = {}
    for k, submesh in triangle_mesh.items():
        cells = submesh.cells["nodes"]
        node_coords = submesh.node_coords

        """
        test

        cells = [np.array([0,1,3]),np.array([1,3,4]),np.array([1,4,2]),np.array([4,5,2]),
                     np.array([3,4,6]),np.array([6,7,4]),np.array([4,5,7]),np.array([5,7,8])]
        node_coords = [np.array([100,100]),np.array([150,100]),np.array([200,100]),
                       np.array([100,150]),np.array([150,150]),np.array([200,150]),
                       np.array([100,200]),np.array([150,200]),np.array([200,200])]
         """

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

        #store cells
        cells = [quad["nodes"] for quad in chosen_quads]

        #store cell quality
        qual = [quad["qual"] for quad in chosen_quads]

        #compute edges
        edges = edges_from_cells(cells)

        #store submesh
        quad_mesh[k] = QuadMesh(node_coords,
                                cells= cells,
                                cell_quality = qual,
                                edges = edges
                                )
    return quad_mesh

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

