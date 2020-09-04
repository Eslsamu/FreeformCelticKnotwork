from shapely.geometry import Polygon
import distmesh
import dmsh
from advancing_front import wavefront_meshing
import optimesh
import meshplex

"""
Mesh each segment and skip segments enclosed in other segments to display object features

returns a dictionary of triangles meshes for each segment
"""
def mesh_segments(segments, edge_size, generator, cvt = False):
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

        if generator == "dmsh":
            geo = polygon2geo(polygon)
            # inspect geo
            #geo.save(str(i) +".png")
            X, cells = distmesh.generate(geo, edge_size=edge_size, verbose = False)
            print("nodes", X.shape, "cells", cells.shape)
        elif generator == "wavefront":
            X, cells = wavefront_meshing(polygon,edge_size)
        else:
            raise Exception("choose dmsh or wavefront as mesh generator")

        if cvt:
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
    print(boundary_e)
    # interior boundary
    inner = []
    for int in polygon.interiors:
        h = list(int.coords.xy)
        boundary_h = [[h[0][i], h[1][i]] for i in range(len(h[0]) - 1)]
        hole = dmsh.Polygon(boundary_h)
        inner.append(hole)

    if len(inner) > 1:
        inner = distmesh.Union(inner)
        geo = dmsh.Difference(outer, inner)
    elif len(inner) == 1:
        inner = inner[0]
        geo = dmsh.Difference(outer, inner)
    else:
        geo = outer
    return geo



