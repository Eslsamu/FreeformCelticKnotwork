import dmsh
import numpy as np

def generate(
        geo,
        edge_size,
        # smoothing_method="distmesh",
        tol=1.0e-5,
        random_seed=0,
        show=False,
        max_steps=10000,
        verbose=False,
):
    # Find h0 from edge_size (function)
    if callable(edge_size):
        edge_size_function = edge_size
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = dmsh.main.create_staggered_grid(h00, geo.bounding_box)
        sizes = edge_size_function(pts.T)
        assert np.all(sizes > 0.0), "edge_size_function must be strictly positive."
        h0 = np.min(sizes)
    else:
        h0 = edge_size

        def edge_size_function(pts):
            return np.full(pts.shape[1], edge_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    pts = dmsh.main.create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / edge_size_function(pts.T) ** 2
    pts = pts[np.random.rand(pts.shape[0]) < alpha / np.max(alpha)]

    num_feature_points = geo.feature_points.shape[0]
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = np.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = np.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = np.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = np.concatenate([geo.feature_points, pts])

    cells, edges = dmsh.main._recell(pts, geo)

    mesh = dmsh.main.meshplex.MeshTri(pts, cells)

    # # move boundary points to the boundary exactly
    # is_boundary_node = mesh.is_boundary_node.copy()
    # mesh.node_coords[is_boundary_node] = geo.boundary_step(
    #     mesh.node_coords[is_boundary_node].T
    # ).T
    # mesh.update_values()

    # print(sum(is_boundary_node))
    # show_mesh(pts, cells, geo)
    # exit(1)

    # if smoothing_method == "odt":
    #     points, cells = optimesh.odt.fixed_point_uniform(
    #         mesh.node_coords,
    #         mesh.cells["nodes"],
    #         max_num_steps=max_steps,
    #         verbose=verbose,
    #         boundary_step=geo.boundary_step,
    #     )
    # else:
    #     assert smoothing_method == "distmesh"
    mesh = dmsh.main.distmesh_smoothing(
        mesh,
        edges,
        geo,
        num_feature_points,
        edge_size_function,
        max_steps,
        tol,
        verbose,
        show,
        delta_t=0.2,
        f_scale=1.2,
    )
    points = mesh.node_coords
    cells = mesh.cells["nodes"]

    return points, cells, edges