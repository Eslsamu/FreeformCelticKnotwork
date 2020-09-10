import numpy as np
from helpers import edges_from_cells,  angle
from shapely.ops import nearest_points
from shapely.geometry import Point
import mip

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

#based on standard deviation of egde lengths or angles
def geometric_irregularity(nodes, mode = "std-angle"):

    if mode == "std-edges":
        dist = [np.sqrt((nodes[i-1][0] - nodes[i][0]) ** 2 + (nodes[i-1][1] - nodes[i][1]) ** 2)
                for i in range(len(nodes))]
        result = 1 - np.std(dist) / np.mean(dist)
    if mode == "std-angle":
        angles = []
        for i in range(len(nodes)):
            # edge as vector
            v1 = nodes[i - 1] - nodes[i]
            v2 = nodes[i] - nodes[(i + 1) % len(nodes)]
            angles.append(angle(v1,v2))
        result = 1 - np.std(angles) / np.pi
        #if angles don't add up to 360 degrees (only for quads) that means their dot product has a different sign somewhere
        #so it is non-convex --> avoid this quad
        if len(nodes)==4 and not np.isclose(np.sum(angles), 2*np.pi):
            return 0
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

        angles.append((angle(v1,v2) % (0.5 * np.pi)))

    return 1 - (np.mean(angles) / ( 0.5 * np.pi))

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

#ILP using the CBC solver to choose the best triangle to quadrilateral conversion
def triangle_to_3_4_ilp(triangle_mesh,  segments,alpha):
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
                    nodes = np.array([dist_node1, shrd_nodes[0], dist_node2, shrd_nodes[1]])
                    # check if this quad was already added
                    if not any(all(node in quad["nodes"] for node in nodes) for quad in possible_quads):
                        coords = [node_coords[node] for node in nodes]
                        #optimal quality is 1, worst is 0
                        qual = alpha * geometric_irregularity(coords) + \
                               (1 - alpha) * directionality_error(coords, segments[k]["polygon"])

                        possible_quads.append({"nodes": nodes, "coords": coords,
                                               "qual": qual, "triangles": [triangle, neighbor]})
            except KeyError:  # skip if triangle has no neighbors
                continue

        #create ILP model
        m = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

        #stores each quad or single triangle as integer variable with its corresponding quality
        var_qual = {}

        #quadrilaterals
        for i,quad in enumerate(possible_quads):
            key_i = "quad "+str(quad["nodes"])
            #create or get variable for each quad
            if key_i not in var_qual.keys():
                x = m.add_var(var_type=mip.BINARY,name=key_i)
                var_qual[key_i] = (x,quad["qual"], quad)
            else:
                x, _, _ = var_qual[key_i]

            #find constraining quads
            for j,const in enumerate(possible_quads):
                if not (quad is const):
                    key_j = "quad "+ str(const["nodes"])
                    if (any(np.array_equal(quad["triangles"][0], t) for t in const["triangles"]))\
                            or (any(np.array_equal(quad["triangles"][1], t) for t in const["triangles"])):
                        #create or get variable for this quad
                        if key_j not in var_qual.keys():
                            y = m.add_var(var_type=mip.BINARY,name=key_j)
                            var_qual[key_j] = (y, const["qual"],const)
                        else:
                            y, _, _ = var_qual[key_j]

                        m += x + y <= 1 # add constraint

        #single triangles
        for triangle in cells:
            key_i = "triangle " + str(triangle)
            x = m.add_var(var_type=mip.BINARY, name=key_i)
            #qual for triangles
            qual = 3/10*(alpha  * geometric_irregularity(coords) + \
                               (1 - alpha) * directionality_error(coords, segments[k]["polygon"]))
            var_qual[key_i] = (x, qual, {"nodes":triangle})

            constraining_vars = []
            #find quads which contain this triangle
            for quad in possible_quads:
                if any(np.array_equal(triangle,t) for t in quad["triangles"]):
                    key = "quad " + str(quad["nodes"])
                    constraining_vars.append(var_qual[key][0])
            #add constraint
            m += x + mip.xsum(v for v in constraining_vars) == 1

        #create objective function adding quality of triangles and of quads
        m += mip.xsum(var_qual["quad "+str(quad["nodes"])][0] * var_qual["quad "+str(quad["nodes"])][1] for i,quad in enumerate(possible_quads)
                      )+ mip.xsum(var_qual["triangle "+str(triangle)][0] * var_qual["triangle "+str(triangle)][1] for triangle in cells)

        m.write("model.lp")
        #optimize
        status = m.optimize(max_seconds=300)
        if status == mip.OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(m.objective_value))
        elif status == mip.OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))

        # store cells
        cells = [v[2]["nodes"] for k,v in var_qual.items() if v[0].x]

        # store cell quality
        qual = [v[1] for k,v in var_qual.items() if v[0].x]

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

    node_coords = [np.array([100, 100]), np.array([150, 100]), np.array([200, 100]),
                   np.array([100, 150]), np.array([150, 150]), np.array([200, 150]),
                   np.array([100, 200]), np.array([150, 200]), np.array([210, 210])]
    cells = Delaunay(node_coords).simplices
    print("cells", cells)
    test = {0: Tri_Quad_Mesh(
        node_coords=node_coords,
        cells=cells
    )}

    res = select_quads(node_coords, cells)
    print("res",res)

if __name__ == '__main__':
    main()










