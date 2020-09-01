from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt

def advancing_front_meshing(seed, stepsize=20, th=4.5):
    front = {"node_coords": seed, "direction": wv_direction(seed), "it": np.zeros(len(seed))}
    boundary = LineString(seed)
    area = Polygon(seed)

    it = 0
    while True:
        current_coords = front["node_coords"][front["it"] == it]
        current_dir = front["direction"][front["it"] == it]
        print(it)

        plt.scatter(x=front["node_coords"][:, 0], y=front["node_coords"][:, 1])
        plt.scatter(x=current_coords[:, 0], y=current_coords[:, 1], color="red", marker="X")

        if len(current_coords) == 0 or it == 8:
            break

        # forward steps
        fd_step, fd_dir = wv_fd(current_coords, current_dir, stepsize)

        # side steps
        left_step, left_dir = wv_left(current_coords, current_dir, stepsize)
        right_step, right_dir = wv_right(current_coords, current_dir, stepsize)

        # new coords
        new_coords = np.concatenate([fd_step, left_step, right_step], axis=0)
        new_dir = np.concatenate([fd_dir, left_dir, right_dir], axis=0)

        # remove new points outside or too close to boundary
        rm = []
        for i, vertex in enumerate(new_coords):
            if boundary.distance(Point(*vertex)) < th or not area.contains(Point(*vertex)):
                rm.append(i)
        new_coords = np.delete(new_coords, rm, axis=0)
        new_dir = np.delete(new_dir, rm, axis=0)

        plt.scatter(x=new_coords[:, 0], y=new_coords[:, 1],color="yellow", marker="o")

        # add new coords to the wavefront
        front["node_coords"] = np.concatenate([front["node_coords"],
                                               new_coords], axis=0)
        front["direction"] = np.concatenate([front["direction"],
                                             new_dir], axis=0)
        front["it"] = np.concatenate([front["it"],
                                      np.ones(len(new_coords)) * it + 1], axis=0)

        i = 0
        # merge nodes which are too close
        while True:
            distances = distance_matrix(front["node_coords"], front["node_coords"])
            np.fill_diagonal(distances, np.inf)
            argmin = np.unravel_index(np.argmin(distances), distances.shape)

            # stop merging when minimum distance between to nodes is less than th
            if distances[argmin[0]][argmin[1]] > th:
                break

            i = i + 1
            a = front["node_coords"][argmin[0]]
            b = front["node_coords"][argmin[1]]
            a_dir = front["direction"][argmin[0]]
            b_dir = front["direction"][argmin[1]]
            a_it = front["it"][argmin[0]]
            b_it = front["it"][argmin[1]]

            # merge the two nodes
            merged_coords = np.mean([a, b], axis=0)
            merged_dir = np.mean([a_dir, b_dir], axis=0)

            # delete the old nodes
            front["node_coords"] = np.delete(front["node_coords"], argmin, axis=0)
            front["direction"] = np.delete(front["direction"], argmin, axis=0)
            front["it"] = np.delete(front["it"], argmin, axis=0)

            # add the merged node (try out what happens when added as this or previous iteration)
            front["node_coords"] = np.append(front["node_coords"], [merged_coords], axis=0)
            front["direction"] = np.append(front["direction"], [merged_dir], axis=0)
            front["it"] = np.append(front["it"], [min(a_it, b_it)], axis=0)

        plt.axis('scaled')
        plt.show()
        it += 1
    return front

#wavefront forward step (90 degrees left unit vector from each point to the next on the contour)
def wv_direction(seed):
    x = seed[:, 0]
    y = seed[:, 1]

    vec_x = np.diff(x,prepend=x[-1])
    rev_vec_y = np.diff(y,prepend=y[-1]) * -1

    distances = np.linalg.norm([vec_x,rev_vec_y],axis=0)

    uv_x = np.divide(vec_x, distances, out= np.zeros_like(vec_x), where=distances != 0) * -1
    uv_y = np.divide(rev_vec_y, distances, out=np.zeros_like(rev_vec_y), where=distances != 0) * -1

    direction = np.column_stack([uv_y, uv_x])
    return direction

#wavefront forward step
#(x2 +- u(y1-y2) * (1/l), y2 +- u(x2-x1) * (1/l)) --> xy3 in right angled, regular triangle
#returns new coordinates and direction
def wv_fd(seed, direction, stepsize):
    x = seed[:, 0]
    y = seed[:, 1]

    new_x = x + direction[:,0] * stepsize
    new_y = y + direction[:,1] * stepsize
    return np.column_stack([new_x, new_y]), direction

#wavefront left step
#returns new coordinates and direction
def wv_left(seed, direction, stepsize):
    x = seed[:, 0]
    y = seed[:, 1]
    uv_x_l = - direction[:,1]
    uv_y_l = direction[:,0]
    new_x = x + uv_x_l * stepsize
    new_y = y + uv_y_l * stepsize
    return np.column_stack([new_x, new_y]), np.column_stack([uv_x_l, uv_y_l])

#wavefront right step
#returns new coordinates and direction
def wv_right(seed, direction, stepsize):
    x = seed[:, 0]
    y = seed[:, 1]

    uv_x_r = direction[:,1]
    uv_y_r = - direction[:,0]

    new_x = x + uv_x_r * stepsize
    new_y = y + uv_y_r * stepsize
    return np.column_stack([new_x, new_y]),np.column_stack([uv_x_r, uv_y_r])

def main():
    from segmentation import regularize_contour
    seed = np.array([[0,0],[0,5],[2,6],[5,6],[5,4],[3,2],[5,1],[5,0]])
    wavefront_init = regularize_contour(seed, 1)

    front=advancing_front_meshing(wavefront_init,1,th=0.6)
    plt.scatter(x=front["node_coords"][:, 0], y=front["node_coords"][:, 1])
    plt.axis('scaled')
    plt.show()
if __name__ == '__main__':
    main()