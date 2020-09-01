import numpy
import scipy.spatial

import meshplex

"""
Simple mesh generator inspired by distmesh
author: nschloe
dmsh : https://github.com/nschloe/dmsh

edit: removed feature points and #TODO added image feature force
"""
#TODO add image feature force

def unique_rows(a):
    # The cleaner alternative `numpy.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    b = numpy.ascontiguousarray(a).view(
        numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
    )
    a_unique, inv, cts = numpy.unique(b, return_inverse=True, return_counts=True)
    a_unique = a_unique.view(a.dtype).reshape(-1, a.shape[1])
    return a_unique, inv, cts

def _recell(pts, geo):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = numpy.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    # Determine edges
    edges = numpy.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = numpy.sort(edges, axis=1)
    edges, _, _ = unique_rows(edges)
    return cells, edges


def create_staggered_grid(h, bounding_box):
    x_step = h
    y_step = h * numpy.sqrt(3) / 2
    bb_width = bounding_box[1] - bounding_box[0]
    bb_height = bounding_box[3] - bounding_box[2]
    midpoint = [
        (bounding_box[0] + bounding_box[1]) / 2,
        (bounding_box[2] + bounding_box[3]) / 2,
    ]
    num_x_steps = int(bb_width / x_step)
    if num_x_steps % 2 == 1:
        num_x_steps -= 1
    num_y_steps = int(bb_height / y_step)
    if num_y_steps % 2 == 1:
        num_y_steps -= 1

    # Generate initial (staggered) point list from bounding box.
    # Make sure that the midpoint is one point in the grid.
    x2 = num_x_steps // 2
    y2 = num_y_steps // 2
    x, y = numpy.meshgrid(
        midpoint[0] + x_step * numpy.arange(-x2, x2 + 1),
        midpoint[1] + y_step * numpy.arange(-y2, y2 + 1),
    )
    # Staggered, such that the midpoint is not moved.
    # Unconditionally move to the right, then add more points to the left.
    offset = (y2 + 1) % 2
    x[offset::2] += h / 2

    out = numpy.column_stack([x.reshape(-1), y.reshape(-1)])

    # add points in the staggered lines to preserve symmetry
    n = 2 * (-(-y2 // 2))
    extra = numpy.empty((n, 2))
    extra[:, 0] = midpoint[0] - x_step * x2 - h / 2
    extra[:, 1] = midpoint[1] + y_step * numpy.arange(-y2 + offset, y2 + 1, 2)

    out = numpy.concatenate([out, extra])
    return out

def generate(
    geo,
    edge_size,
    tol=1.0e-5,
    random_seed=0,
    show=False,
    max_steps=1000,
    verbose=False,
):
    # Find h0 from edge_size (function)
    if callable(edge_size):
        edge_size_function = edge_size
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        sizes = edge_size_function(pts.T)
        assert numpy.all(sizes > 0.0), "edge_size_function must be strictly positive."
        h0 = numpy.min(sizes)
    else:
        h0 = edge_size

        def edge_size_function(pts):
            return numpy.full(pts.shape[1], edge_size)

    if random_seed is not None:
        numpy.random.seed(random_seed)

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / edge_size_function(pts.T) ** 2
    pts = pts[numpy.random.rand(pts.shape[0]) < alpha / numpy.max(alpha)]

    num_feature_points = geo.feature_points.shape[0]
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = numpy.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = numpy.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = numpy.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = numpy.concatenate([geo.feature_points, pts])

    cells, edges = _recell(pts, geo)

    mesh = meshplex.MeshTri(pts, cells)

    mesh = distmesh_smoothing(
        mesh,
        edges,
        geo,
        edge_size_function,
        max_steps,
        tol,
        verbose,
        delta_t=0.2,
        f_scale=1.2,
    )

    points = mesh.node_coords
    cells = mesh.cells["nodes"]

    return points, cells


def distmesh_smoothing(
    mesh,
    edges,
    geo,
    edge_size_function,
    max_steps,
    tol,
    verbose,
    delta_t,
    f_scale,
):
    k = 0
    pts_old_last_recell = mesh.node_coords.copy()
    while True:
        if verbose:
            print(f"step {k}")

        if k > max_steps:
            if verbose:
                print(f"Exceeded max_steps ({max_steps}).")
            break

        k += 1

        diff = mesh.node_coords - pts_old_last_recell
        move2_last_recell = numpy.einsum("ij,ij->i", diff, diff)
        if numpy.any(move2_last_recell > 1.0e-2 ** 2):
            pts_old_last_recell = mesh.node_coords.copy()
            cells, edges = _recell(mesh.node_coords, geo)
            #
            mesh = meshplex.MeshTri(mesh.node_coords, cells)

        edges_vec = mesh.node_coords[edges[:, 1]] - mesh.node_coords[edges[:, 0]]
        edge_lengths = numpy.sqrt(numpy.einsum("ij,ij->i", edges_vec, edges_vec))
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

        pts_before_update = mesh.node_coords.copy()

        mesh.node_coords += update


        is_outside = geo.dist(mesh.node_coords.T) > 0.0
        idx = is_outside

        mesh.node_coords[idx] = geo.boundary_step(mesh.node_coords[idx].T).T

        diff = mesh.node_coords - pts_before_update
        move2 = numpy.einsum("ij,ij->i", diff, diff)

        if verbose:
            print("max_move: {:.6e}".format(numpy.sqrt(numpy.max(move2))))

        if numpy.all(move2 < tol ** 2):
            break

    return mesh
