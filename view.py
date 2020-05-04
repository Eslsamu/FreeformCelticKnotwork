import turtle
from mercat import compute_curves
import numpy as np
from mesh_init import initial_lattice
from shapely.geometry import Polygon

click_radius = 10
graph = {"vertices" :[], "edges":[]}
selected_vertex = None


def turtle_cubic_bezier(t,p0, p1, p2, p3, n=20, fill="white", border="black", curvesize=20):
    t.pensize(curvesize)
    t.pencolor(border)
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    t.penup()
    t.goto(x0, y0)
    t.pendown()
    for i in range(n + 1):
        s = i / n
        a = (1. - s) ** 3
        b = 3. * s * (1. - s) ** 2
        c = 3.0 * s ** 2 * (1.0 - s)
        d = s ** 3

        x = int(a * x0 + b * x1 + c * x2 + d * x3)
        y = int(a * y0 + b * y1 + c * y2 + d * y3)
        #creates pen fill/border effect
        lastpos = t.pos()
        t.goto(x, y)
        t.pu()
        t.seth(t.towards(lastpos))
        t.fd(t.distance(lastpos)+5)
        t.pd()
        color = t.pencolor()
        t.pencolor(fill)
        t.pensize(t.pensize()-5)
        t.goto(x,y)
        t.pencolor(color)
        t.pensize(t.pensize()+5)
    t.goto(x3, y3)
    t.pu()

def stepturtle(turtle, x, y):
    """step turtle so it ends at x/y"""

    while turtle.distance(x, y)>0.5:
        rotdif = turtle.towards(x, y)  # see how much we need to turn
        turn = rotdif - turtle.heading()
        if turn > 180:
            turtle.left(turn-360)
        elif turn < -180:
            turtle.left(turn+360)
        else:
            turtle.left(turn)  # turn

        distance = turtle.distance(x, y)  # see how far we need to go
        turtle.forward(distance)  # go

def draw_vertex(t,x,y):
    t.pu()
    t.goto(x,y)
    t.dot(10,"black")
    t.home()

def draw_line(t,v1,v2):
    t.pu()
    t.goto(v1[0],v1[1])
    t.pd()
    t.goto(v2[0],v2[1])
    t.pu()
    t.home()

def draw_arrow(t,v1,v2):
    t.pu()
    t.goto(v1[0], v1[1])
    t.pd()
    angle = t.towards(v2[0], v2[1])
    t.seth(angle)
    t.fd(10)
    t.dot("blue")
    t.goto(v2[0], v2[1])
    t.pu()
    t.home()

def draw_cross(t,x,y,angle):
    t.pu()
    t.goto(x,y)
    t.pd()
    #1
    t.seth(angle+60)
    t.fd(10)
    t.bk(10)
    #2
    t.seth(angle + 120)
    t.fd(10)
    t.bk(10)
    #3
    t.seth(angle -60)
    t.fd(10)
    t.bk(10)
    #4
    t.seth(angle - 120)
    t.fd(10)
    t.bk(10)
    t.pu()
    t.home()

def draw_graph(t,graph):

    vertices = graph["vertices"]
    edges = graph["edges"]
    nodes = []  # store midpoint nodes to connect them later
    t.color("black")
    t.pensize(5)
    #draw vertices
    for v in vertices:
        # draw vertices
        x = v[0]
        y = v[1]
        draw_vertex(t,x,y)
   #draw edges
    for e in edges:
        #draw edge
        v1 = vertices[e[0]]
        v2 = vertices[e[1]]
        draw_arrow(t,v1,v2)



def main():
    t = turtle.Turtle()
    s = turtle.Screen()
    t.speed(0)
    t.ht()
    s.delay(0)
    t.width(5)
    turtle.tracer(0)
    turtle.setworldcoordinates(0,0,1000,500)

    def handle_left_click(x, y):
        global selected_vertex
        global click_radius
        clicked_vertex = [i for i, v in enumerate(graph["vertices"]) if
                          (x - click_radius < v[0] < x + click_radius) and
                          (y - click_radius < v[1] < y + click_radius)]
        if selected_vertex is not None:
            if len(clicked_vertex) == 1:
                new_edge = (selected_vertex, clicked_vertex[0])
                if new_edge not in graph["edges"] and new_edge[0] != new_edge[1]:
                    graph["edges"].append(new_edge)
                    print("added edge")
            else:
                selected_vertex = None
                print("deselect vertex")
        else:
            if len(clicked_vertex) == 1:
                selected_vertex = clicked_vertex[0]
                print("selected vertex", selected_vertex)
            else:
                graph["vertices"].append((x, y))
                print("add vertex")
        t.clear()
        draw_graph(t, graph)

        if selected_vertex:
            t.goto(graph["vertices"][selected_vertex])
            t.dot(10, "red")

    def handle_right_click(x, y):
        global selected_vertex
        global click_radius
        clicked_vertex = [i for i, v in enumerate(graph["vertices"]) if
                          (x - click_radius < v[0] < x + click_radius) and
                          (y - click_radius < v[1] < y + click_radius)]
        # delete vertex and adjacent edges
        if len(clicked_vertex) == 1:
            print("delete vertex", clicked_vertex[0])
            graph["vertices"].pop(clicked_vertex[0])
            graph["edges"] = [edge for edge in graph["edges"] if clicked_vertex[0] not in edge]

            # shift index by one after this vertex for all other edges
            for i, edge in enumerate(graph["edges"]):
                v1, v2 = edge[0], edge[1]
                if v1 > clicked_vertex[0]:
                    v1 = v1 - 1
                if v2 > clicked_vertex[0]:
                    v2 = v2 - 1
                graph["edges"][i] = (v1, v2)
            if selected_vertex:
                if selected_vertex > clicked_vertex[0]:
                    selected_vertex = selected_vertex - 1
        else:
            if selected_vertex:
                graph["vertices"][selected_vertex] = (x, y)
                print("move vertex")

        t.clear()
        draw_graph(t, graph)

    def squish_inc():
        global squish
        squish = squish + 0.1
        t.clear()
        # draw_graph(t, graph)
        # draw_curves(nodes, graph["vertices"], graph["edges"])

    def squish_dec():
        global squish
        squish = squish - 0.1
        t.clear()
        # draw_graph(t, graph)
        # draw_curves(nodes, graph["vertices"], graph["edges"])

    def inc_size():
        global curvesize
        curvesize = curvesize + 1
        t.clear()
        # draw_graph(t, graph)
        # draw_curves(nodes, graph["vertices"], graph["edges"])

    def dec_size():
        global curvesize
        curvesize = curvesize - 1
        t.clear()
        # draw_graph(t, graph)
        # draw_curves(nodes, graph["vertices"], graph["edges"])

    def draw_knotwork(show_points=True):
        curves = compute_curves(graph["vertices"], graph["edges"])
        for p0, p1, p2, p3 in curves:
            turtle_cubic_bezier(t, p0, p1, p2, p3)
            if show_points:
                # draw control point area
                t.color("green")
                t.pensize(1)
                t.pd()
                t.goto(p0)
                t.goto(p1)
                t.dot("purple")
                t.goto(p2)
                t.dot("orange")
                t.goto(p3)
                t.pu()

    def draw_graph_fill():
        boundary = Polygon(graph["vertices"])
        print(boundary)
        draw_graph(t,initial_lattice(25,boundary))
        print("OK")


    turtle.onscreenclick(handle_left_click, btn=1)
    turtle.onscreenclick(handle_right_click, btn=3)

    s.onkey(squish_inc, "Up")
    s.onkey(squish_dec, "Down")
    s.onkey(inc_size, "Right")
    s.onkey(dec_size, "Left")
    s.onkey(draw_knotwork, "space")
    s.onkey(draw_graph_fill, "a")
    s.listen()
    #s.bgpic("../eyeball.gif")
    turtle.mainloop()

if __name__ == '__main__':
    main()
