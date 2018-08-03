import face_recognition
import math

def get_point_by_id(landmark_object, num):
    if num <= 17:
        location = "chin"
        offset = num
    elif num <= 22:
        location = "left_eyebrow"
        offset = num - 17
    elif num <= 27:
        location = "right_eyebrow"
        offset = num - 22
    elif num <= 31:
        location = "nose_bridge"
        offset = num - 27
    elif num <= 36:
        location = "nose_tip"
        offset = num - 31
    elif num <= 42:
        location = "left_eye"
        offset = num - 36
    elif num <= 48:
        location = "right_eye"
        offset = num - 42
    elif num <= 60:
        location = "top_lip"
        offset = num - 48
    elif num <= 72:
        location = "bottom_lip"
        offset = num - 60
    # returns a tuple of coordinates
    return(landmark_object[0][location][offset - 1])

def get_points_by_ID(landmark_object, list_of_id_tuples):
    end_point_list = []
    for tuple in list_of_id_tuples:
        a, b = tuple
        A = get_point_by_id(landmark_object, a)
        B = get_point_by_id(landmark_object, b)
        # Compatibility with matplotlib
        x1, y1 = A
        x2, y2 = B
        X = (x1, x2)
        Y = (y1, y2)
        end_point_list.append([X, Y])
    return end_point_list

# returns a tuple of of x and y coordinates lists
def get_coord_list(landmark_object):
    x_coordinates = []
    y_coordinates = []
    for face in landmark_object:
        for part in face:
            for point in face[part]:
                x, y = point
                x_coordinates.append(x)
                y_coordinates.append(y)
    return (x_coordinates, y_coordinates)

def get_distance(list_of_coordinates):
    distance_list = []
    for element in list_of_coordinates:
        dis = euclidean_distance(element[0], element[1])
        distance_list.append(dis)
    return distance_list

def get_midpoint(list_of_coordinates):
    midpoint_list = []
    for element in list_of_coordinates:
        mid = midpoint(element[0], element[1])
        midpoint_list.append(mid)
    return midpoint_list

def euclidean_distance(p1, p2):
    # gives the eucildean distance between two points
    x1, x2 = p1
    y1, y2 = p2
    euc_dis = math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    euc_dis = round(euc_dis, 2)
    return euc_dis

def midpoint(p1, p2):
    # gives midpoint of two points
    x1, x2 = p1
    y1, y2 = p2
    X = (x1 + x2) / 2
    Y = (y1 + y2) / 2
    middle = (X, Y)
    return middle
