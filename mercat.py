import numpy as np

def midpoint(v1,v2):
    return (v1[0]+v2[0])/2 , (v1[1]+v2[1])/2

"""
angle starts counterclockwise from west to east in turtle
"""
def compute_curves(vertices,edges,nodes = None,curves =[],squish=0.5):
    if nodes == None:
        nodes = []
        curves = []
        for e in edges:
            nodes.extend(((e, np.pi * 3/4), (e, np.pi * 7/4), (e, np.pi / 4), (e, np.pi * 5/4)))
    #recursive function stopping condition
    if len(nodes) == 0:
        return curves

    curr_node = nodes.pop(0)
    curr_edge = curr_node[0]
    curr_dir = curr_node[1]

    #determine which junction the node faces
    if curr_dir < np.pi/2  or curr_dir > 3*np.pi/2:
        facing_junction = curr_edge[0]
        opp_junction = curr_edge[1]
    else:
        facing_junction = curr_edge[1]
        opp_junction = curr_edge[0]


    #get all distinct edges on this junction/vertex
    adj_edges = [edge for edge in edges if (facing_junction in edge) & (edge != curr_edge)]
    #print("curr edge facing junc and adj edges curr dir",curr_edge, facing_junction,adj_edges, curr_dir)

    junc_x,junc_y = vertices[facing_junction]
    opp_x, opp_y = vertices[opp_junction]

    #angles of adjacent edges
    angles = []

    #case in which there are no other adjacent edges to the junction
    if len(adj_edges) == 0:
        next_edge = curr_edge
    else:
        for adj_e in adj_edges:
            v2 = vertices[[v for v in adj_e if v != facing_junction][0]]
            # shift the coordinate system to have the junction as the origin for vector based calculations
            v1_x = opp_x - junc_x
            v1_y = opp_y - junc_y
            v2_x = v2[0] - junc_x
            v2_y = v2[1] - junc_y
            #compute the angle of the two vectors
            dot = v1_x*v2_x + v1_y*v2_y
            det = v1_x*v2_y - v1_y*v2_x
            angle = np.arctan2(det,dot)
            if angle < 0:
                angle += 2*np.pi
            #print("angle/edge", angle * nodes / np.pi, adj_e)
            #take the clockwise angle if nodes are on the left side facing the junction
            if curr_dir == np.pi/4 or curr_dir == np.pi*5/4:
                angle = 2 * np.pi - angle

            angles.append(angle)
            #print("angle converted", angle*180/np.pi)

        i_shortest_angle = np.argmin(angles)
        next_edge = adj_edges[i_shortest_angle]

        #if the shortest angle is less than 180 degrees, the nodes face each other
        #TODO this isn't right
        facing_nodes =  False
        if angles[i_shortest_angle] < 0:
            facing_nodes = True


    #determine the direction of the next edge
    #if both edges face the same direction then
    if facing_junction == next_edge[0]:
        #take the node of the other edge's opposite side of the same node direction
        if curr_dir == np.pi/4 or curr_dir == np.pi*5/4:
            next_dir = np.pi*7/4
        elif curr_dir == np.pi*3/4 or curr_dir == np.pi*7/4:
            next_dir = np.pi/4
    else: #else take the node of the same side for the opposite direction
        if curr_dir == np.pi/4 or curr_dir == np.pi*5/4:
            next_dir = np.pi*3/4
        elif curr_dir == np.pi*7/4 or curr_dir == np.pi*3/4:
            next_dir = np.pi*5/4


    next_node = [node for node in nodes if (next_edge == node[0]) & (next_dir == node[1])][0]

    #determine anchor points of bezier curve
    x1, y1 = midpoint(vertices[curr_edge[0]], vertices[curr_edge[1]])
    x2, y2 = midpoint(vertices[next_edge[0]], vertices[next_edge[1]])

    #squishing parameter for nodes facing apart from each other
    if len(angles) > 0:
        tau = (np.min(angles)/np.pi) * squish
    else:
        tau = squish/2

    next_edge_opp_x, next_edge_opp_y = vertices[[v for v in next_edge if v != facing_junction][0]]
    edgelength1 = np.sqrt((opp_x - junc_x) ** 2 + (opp_y - junc_y) ** 2)
    edgelength2 = np.sqrt((next_edge_opp_x - junc_x) ** 2 + (next_edge_opp_y - junc_y) ** 2)

    if (curr_edge == next_edge):
        #curve 1
        #node 1 is the current node and node 2 is at facing vertex
        #cntr1 is going in the usual way in current direction by tau (?)
        #cntrl2 is going up 90 degrees with (distance/2)^2-distance(midpoint,cntrl1)^2

        edge_angle = np.arctan2(y1-vertices[curr_edge[1]][1],x1-vertices[curr_edge[1]][0])


        cntrl1 = [x1 + np.cos(curr_dir + edge_angle) * edgelength1 * tau,
                 y1 + np.sin(curr_dir + edge_angle) * edgelength1 * tau]

        edgelength2 = np.arccos(np.pi/4) * edgelength1

        if curr_dir == np.pi/4 or curr_dir == np.pi*7/4:
            dir_angle = -np.pi/2
        else:
            dir_angle = np.pi/2

        cntrl2 = [junc_x + np.cos(edge_angle + dir_angle) * edgelength2 * tau,
                 junc_y + np.sin(edge_angle + dir_angle) * edgelength2 * tau]

        curves.append(([x1, y1], cntrl1, cntrl2, vertices[facing_junction]))

        # curve 2
        edge_angle = np.arctan2(y1 - vertices[curr_edge[1]][1], x1 - vertices[curr_edge[1]][0])

        cntrl1 = [x1 + np.cos(next_dir + edge_angle) * edgelength1 * tau,
                  y1 + np.sin(next_dir + edge_angle) * edgelength1 * tau]

        adjacent_length = np.arccos(np.pi / 4) * edgelength1

        if curr_dir == np.pi/4 or curr_dir == np.pi*7/4:
            dir_angle =  -np.pi/2
        else:
            dir_angle = np.pi/2

        cntrl2 = [junc_x + np.cos(edge_angle - dir_angle) * adjacent_length * tau,
                  junc_y + np.sin(edge_angle - dir_angle) * adjacent_length * tau]

        curves.append(([x1, y1], cntrl1, cntrl2, vertices[facing_junction]))
    else:
        edge_angle1 = np.arctan2(y1 - vertices[curr_edge[1]][1], x1 - vertices[curr_edge[1]][0])

        cntrl1 = [x1 + np.cos(curr_dir + edge_angle1) * edgelength1 * tau,
                  y1 + np.sin(curr_dir + edge_angle1) * edgelength1 * tau]

        edge_angle2 = np.arctan2(y2 - vertices[next_edge[1]][1], x2 - vertices[next_edge[1]][0])

        cntrl2 = [x2 + np.cos(next_dir + edge_angle2) * edgelength2 * tau,
                  y2 + np.sin(next_dir + edge_angle2) * edgelength2 * tau]

        curves.append(([x1, y1], cntrl1, cntrl2, [x2,y2]))

    #recursive call of this method starting with the next node
    nodes.pop(nodes.index(next_node))
    return compute_curves(vertices, edges,nodes, curves)






















#TODO
#

#mesh generation


#inflating curves (with or without culling overlapping curves?(can be done later))

#overlaps?


#reduce undulations (look at both kaplan and context free paper)

#introducing breakpoints



#method of master student paper
"""
    else:
        t.goto(x1,y1)
        dist = t.distance(x2,y2)
        beta = t.towards(x2,y2)
        t.seth(beta)
        alpha1 = t.towards(junc_x,junc_y) + curr_dir
        t.fd(dist/gam)
        t.seth(beta + np.pi/2)
        #t.fd(np.arctan(alpha1)*dist / gam) #TODO remove undulations here
        anch1 = t.pos()

        t.goto(x2,y2)
        beta2 = t.towards(x1, y1)
        t.seth(beta2)
        alpha2 = t.towards(junc_x, junc_y) + next_dir
        t.fd(dist / gam)
        t.seth(beta2 - np.pi/2)
        #t.bk(np.arctan(alpha2) * dist / gam)
        anch2 = t.pos()

        #print(beta,beta2, curr_dir, next_dir)
    """