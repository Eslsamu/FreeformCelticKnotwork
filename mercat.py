import numpy as np

def midpoint(v1,v2):
    return (v1[0]+v2[0])/2 , (v1[1]+v2[1])/2


def get_node_order(vertices, edges, nodes):
    current_thread = 0
    sorted_threads = {current_thread:[]}
    next_start_node = nodes.pop(0)
    while True:
        if len(nodes) == 0:
            return sorted_threads

        #print(len(sorted_nodes),len(nodes))
        start_node = next_start_node
        start_edge = start_node[0]
        start_dir = start_node[1]

        # determine which junction the node faces
        if start_node[2]:
            facing_junction = start_edge[0]
            opp_junction = start_edge[1]
        else:
            facing_junction = start_edge[1]
            opp_junction = start_edge[0]

        # get all adjecent edges on this junction/vertex
        adj_edges = [edge for edge in edges if (facing_junction in edge) & (edge != start_edge)]

        junc_x, junc_y = vertices[facing_junction]
        opp_x, opp_y = vertices[opp_junction]

        # angles of adjacent edges
        angles = []

        # in case there are no other adjacent edges to the junction -> curve to the same edge
        if len(adj_edges) == 0:
            end_edge = start_edge
        # else curve to next adjacent edge
        else:
            for adj_e in adj_edges:
                v2 = vertices[[v for v in adj_e if v != facing_junction][0]]
                # shift the coordinate system to have the junction as the origin for vector based calculations
                v1_x = opp_x - junc_x
                v1_y = opp_y - junc_y
                v2_x = v2[0] - junc_x
                v2_y = v2[1] - junc_y
                # compute the angle of the two vectors
                dot = v1_x * v2_x + v1_y * v2_y
                det = v1_x * v2_y - v1_y * v2_x
                angle = np.arctan2(det, dot)
                if angle < 0:
                    angle += 2 * np.pi
                # take the clockwise angle if nodes are on the left side facing the junction
                if start_dir == np.pi / 4 or start_dir == np.pi * 5 / 4:
                    angle = 2 * np.pi - angle
                angles.append(angle)

            i_shortest_angle = np.argmin(angles)
            end_edge = adj_edges[i_shortest_angle]
        #print("adj", adj_edges)
        #print("angles", angles)
        # determine the direction of the next edge
        # if both edges face the same direction then
        if facing_junction == end_edge[0]:
            # take the node of the other edge's opposite side of the same node direction
            if np.isclose(start_dir, np.pi / 4, atol=0.1)  or np.isclose(start_dir, np.pi * 5 / 4,atol=0.1):
                end_dir = np.pi * 7 / 4
            elif  np.isclose(start_dir, np.pi * 3 / 4,atol=0.1) or np.isclose(start_dir, np.pi * 7 / 4,atol=0.1) :
                end_dir = np.pi / 4
        else:  # else take the node of the same side for the opposite direction
            if  np.isclose(start_dir, np.pi / 4,atol=0.1)  or  np.isclose(start_dir, np.pi * 5 / 4,atol=0.1):
                end_dir = np.pi * 3 / 4
            elif  np.isclose(start_dir, np.pi * 7 / 4,atol=0.1)  or  np.isclose(start_dir, np.pi * 3 / 4,atol=0.1) :
                end_dir = np.pi * 5 / 4

        try:
            end_node_i = [j for j in range(len(nodes)) if (end_edge == nodes[j][0]) & (end_dir == nodes[j][1])][0]
            end_node = nodes.pop(end_node_i)
            #store angle between two nodes
            end_node[3] = np.min(angles)
            start_node[3] = end_node[3]
        except IndexError:
            #start a new thread if the next node was already used
            next_start_node = nodes.pop(0)
            current_thread += 1
            sorted_threads[current_thread] = []
            continue

        #print("start",start_node)
        #print("end", end_node)
        sorted_threads[current_thread].append(start_node)
        sorted_threads[current_thread].append(end_node)

        next_start_edge = end_edge
        next_start_dir = (end_dir + np.pi) % (2*np.pi)
        next_facing = not end_node[2]

        next_start_node = [next_start_edge,next_start_dir,next_facing, None]

        #remove next starting node from list
        for n in nodes:
            if n[0] == end_edge:
                if np.isclose(n[1],next_start_dir,atol=0.1):
                    nodes.remove(n)
                    break


"""
Computing bezier curves on a graph using the Mercat Algorithm
(angle starts counterclockwise)
"""
def compute_curves(vertices, edges, squish=0.5):
    nodes = []
    for e in edges:
        #(edge - node direction - node directed to first edge vertex - angle to next node)
        nodes.extend([[e, np.pi * 3 / 4, False,None], [e, np.pi * 7 / 4,True, None],
                      [e, np.pi / 4, True,None], [e, np.pi * 5 / 4, False,None]])

    #sorts threads in the right interlacement order
    sorted_threads = get_node_order(vertices, edges, nodes)
    curves = {}
    for k, thread in sorted_threads.items():
        i = 0
        curves[k] = []
        while True:
            i += 1
            #print(i)
            curr_node = thread.pop(0)
            curr_edge = curr_node[0]
            curr_dir = curr_node[1]
            curr_angle = curr_node[3]
            #print("curr node", curr_node)


            next_node = thread.pop(0)
            next_edge = next_node[0]
            next_dir = next_node[1]
            #print("next node", next_node)

            # determine which junction the node faces
            if curr_node[2]:
                facing_junction = curr_edge[0]
                opp_junction = curr_edge[1]
            else:
                facing_junction = curr_edge[1]
                opp_junction = curr_edge[0]

            junc_x, junc_y = vertices[facing_junction]
            opp_x, opp_y = vertices[opp_junction]

            # determine anchor points of bezier curve
            x1, y1 = midpoint(vertices[curr_edge[0]], vertices[curr_edge[1]])
            x2, y2 = midpoint(vertices[next_edge[0]], vertices[next_edge[1]])

            next_edge_opp_x, next_edge_opp_y = vertices[[v for v in next_edge if v != facing_junction][0]]
            edgelength1 = np.sqrt((opp_x - junc_x) ** 2 + (opp_y - junc_y) ** 2)
            edgelength2 = np.sqrt((next_edge_opp_x - junc_x) ** 2 + (next_edge_opp_y - junc_y) ** 2)

            # squishing parameter for nodes facing apart from each other
            if curr_angle > 0:
                tau = (curr_angle / np.pi) * squish
            else:
                tau = squish / 2


            #case that curve stays on the same edge -> create two curves meeting at the end of the edge
            if (curr_edge == next_edge):
                #curve 1
                edge_angle = np.arctan2(y1 - vertices[curr_edge[1]][1], x1 - vertices[curr_edge[1]][0])

                cntrl1 = [x1 + np.cos(curr_dir + edge_angle) * edgelength1 * tau,
                          y1 + np.sin(curr_dir + edge_angle) * edgelength1 * tau]

                edgelength2 = np.arccos(np.pi / 4) * edgelength1

                if curr_dir == np.pi / 4 or curr_dir == np.pi * 7 / 4:
                    dir_angle = -np.pi / 2
                else:
                    dir_angle = np.pi / 2

                cntrl2 = [junc_x + np.cos(edge_angle + dir_angle) * edgelength2 * tau,
                          junc_y + np.sin(edge_angle + dir_angle) * edgelength2 * tau]

                curves[k].append(([x1, y1], cntrl1, cntrl2, vertices[facing_junction],curr_dir))

                # curve 2
                edge_angle = np.arctan2(y1 - vertices[curr_edge[1]][1], x1 - vertices[curr_edge[1]][0])

                cntrl1 = [x1 + np.cos(next_dir + edge_angle) * edgelength1 * tau,
                          y1 + np.sin(next_dir + edge_angle) * edgelength1 * tau]

                adjacent_length = np.arccos(np.pi / 4) * edgelength1

                if curr_dir == np.pi / 4 or curr_dir == np.pi * 7 / 4:
                    dir_angle = -np.pi / 2
                else:
                    dir_angle = np.pi / 2

                cntrl2 = [junc_x + np.cos(edge_angle - dir_angle) * adjacent_length * tau,
                          junc_y + np.sin(edge_angle - dir_angle) * adjacent_length * tau]
                #print(([x1, y1], cntrl1, cntrl2, vertices[facing_junction]))
                curves[k].append(([x1, y1], cntrl1, cntrl2, vertices[facing_junction],next_dir))
            else: #normal case
                edge_angle1 = np.arctan2(y1 - vertices[curr_edge[1]][1], x1 - vertices[curr_edge[1]][0])

                cntrl1 = [x1 + np.cos(curr_dir + edge_angle1) * edgelength1 * tau,
                          y1 + np.sin(curr_dir + edge_angle1) * edgelength1 * tau]
                #print(tau)

                edge_angle2 = np.arctan2(y2 - vertices[next_edge[1]][1], x2 - vertices[next_edge[1]][0])

                cntrl2 = [x2 + np.cos(next_dir + edge_angle2) * edgelength2 * tau,
                          y2 + np.sin(next_dir + edge_angle2) * edgelength2 * tau]

                #print(edge_angle1,edge_angle2)
                #print(([x1, y1], cntrl1, cntrl2, [x2, y2]))
                curves[k].append(([x1, y1], cntrl1, cntrl2, [x2, y2],curr_dir))

            if len(thread) == 0:
                break
            #print("---------------")

    return curves





