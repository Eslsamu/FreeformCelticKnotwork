from shapely.ops import polylabel
from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import numpy as np
import cv2 as cv
from simplification.cutil import (
        simplify_coords_vwp,
    )

#a segment is represented as a polygon
def contours_to_segments(contours, min_diameter):
    segments = []
    thin_segments = []
    for i in range(len(contours)):
        contour1 = contours[i]

        #find all contours inside inside the current contour
        contains = []
        for j in range(len(contours)):
            if contour1.contains(contours[j]) and i != j:
                contains.append(contours[j])

        #create a segment of the current contour with its containing contours as holes
        segment = contour1
        for j, cnt in enumerate(contains):
            # inspect segment
            segment = segment.difference(cnt)

        # find visual center (polylabel) of segment
        label = polylabel(segment, tolerance=1)
        # determine distance from segment border
        dist = segment.exterior.distance(Point(label.x, label.y))
        # remove segment if the distance from its center is less than an edge size (too thin)
        if dist < min_diameter:
            # print("too small contour", i, contour)
            thin_segments.append({"polygon":segment, "contains": contains, "diameter":dist, "polylabel": label})
        else:
            segments.append({"polygon":segment, "contains": contains, "diameter":dist,"polylabel": label})

    return segments, thin_segments



#
def regularize_contour(points, min_length):
    points = np.array(points,dtype=np.float64)

    # independent vector for x and y
    x = points[:, 0]
    y = points[:, 1]

    # close the contour temporarily
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])

    # distance between consecutive points
    dx = np.diff(xc)
    dy = np.diff(yc)
    dS = np.sqrt(dx ** 2 + dy ** 2)


    new_points = []
    for i in range(len(points)):
        ax = xc[i]
        ay = yc[i]

        bx = xc[(i+1) % len(points)]
        by = yc[(i+1) % len(points)]

        segments = int(dS[i] / min_length)
        if segments == 0:
            new_points.append([bx,by])
        for s in range(segments):
            new_x = ax + (bx-ax) * (s+1) / segments
            new_y = ay + (by-ay) * (s+1) / segments
            new_points.append([new_x,new_y])
    return np.array(new_points)

"""
"""
def findContours(edges, min_arclength, contour_roughness):

    contour_points, h = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("inital amount of cnts", len(contour_points))

    #remove all contours with arclength less than 3 * edge length
    cnts = []
    for contour in contour_points:
        if cv.arcLength(contour,True) > min_arclength * 3:
            cnts.append(contour)
    print("removed", len(contour_points) - len(cnts), "short contours")
    contour_points = cnts

    #simplify polygons using a topology preserving variant of the Visvalingam-Whyatt Algorithm
    contour_points = [simplify_coords_vwp(np.squeeze(contour), contour_roughness) for contour in contour_points]

    # delete contours with less than 3 points (not even forming a polygon)
    cnts = []
    for contour in contour_points:
        if len(contour) > 2:
            cnts.append(contour)
    print("removed", len(contour_points) - len(cnts), " contours with less than 3 points")
    contour_points = cnts

    #turn contours into polygon objects
    contour_points = [Polygon(contour) for contour in contour_points]

    #delete invalid contours (non-closing)
    cnts = []
    invalid_contours = []
    for polygon in contour_points:
        if polygon.is_valid:
            cnts.append(polygon)
        else:
            invalid_contours.append(polygon)
    print("removed", len(contour_points) - len(cnts), " contours that were invalid")
    contour_points = cnts

    #delete overlapping contours
    cnts = []
    overlap_cnts = []
    for i in range(len(contour_points)):
        a = contour_points[i]
        if a not in overlap_cnts:
            cnts.append(a)
        for j in range(i,len(contour_points)):
            b = contour_points[j]
            if a != b:
                if a.overlaps(b):
                    overlap_cnts.append(b)

    print("removed", len(contour_points) - len(cnts), " contours that were overlapping")
    contour_points = [cnt for cnt in cnts if cnt not in overlap_cnts]

    print("found", len(contour_points), "valid contours")

    return contour_points, invalid_contours

"""
"""
def find_edges(img_file, canny_th, blur):
    im = cv.imread(img_file)
    # resize image (500x500 is ideal)
    # scale = 250000 / (im.shape[0] * im.shape[1])
    # height = int(im.shape[0] * scale)
    # width = int(im.shape[1] * scale)
    # im = cv.resize(im, (width,height), interpolation = cv.INTER_AREA)

    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    median = np.median(imgray)
    thresh = median * canny_th
    edges = cv.Canny(im, thresh, thresh * 2)

    # if blur is 1 there is no effect
    edges = cv.blur(edges, (blur, blur))
    return edges

"""
"""
def run_segmentation(img_file, canny_th, blur, min_diameter, min_arclength, contour_roughness):
    #process image
    edges = find_edges(img_file, canny_th, blur)

    #find contours
    contours, invalid = findContours(edges, min_arclength, contour_roughness)

    #form segments from contours
    segments, thin_segments = contours_to_segments(contours, min_diameter)

    print("found",len(segments),"valid segments")

    return segments, thin_segments, invalid, edges

