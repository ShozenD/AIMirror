import cv2
import modules.LandPoint as LandPoint
import numpy as np
import face_recognition

def mask_hdif(hdif_results: list):
    """
    return hdif alpha masks 
    hdif_results: the results of the hdif analysis in a list
    color: the color channels we want to display
    """

    # Prepare Alpha
    alpha_list = []
    for hdif in hdif_results:                          
        hdif = hdif.astype(float)/255 
        alpha_list.append(cv2.merge((hdif,hdif,hdif)))

    # return result
    return alpha_list

def mask_hessian(hessian_results: list):
    """
    background: The image we want to draw over
    hdif_results: the results of the hdif analysis in a list
    color: the color channels we want to display
    """

    # Prepare Alpha                         
    hhf = hessian_results.astype(float)
    alpha = cv2.merge((hhf,hhf,hhf))

    # return result
    return alpha

def face_polygon(image: list, point_id: list):
    """
    Creates a cropped out image of the desired polygon.
    Image is the return value from the cv2.imread function
    Point_id is the point id of the polygon vertices
    """
    # Initialize Variables
    coord_list = []
    blank = np.zeros(image.shape, float) 

    # Obtain Facial Landmarks 
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    # Create Coodinate Points 
    for point in point_id:
        coord_list.append(LandPoint.get_point_by_id(face_landmarks_list, point))

    # List of Polygon Vertices
    vertices = np.array(coord_list)

    mask = cv2.fillConvexPoly(blank, vertices, (1,1,1))

    # Background 
    fg = image.astype(float)
    # Foreground 
    bg = blank.astype(float)
    # Alpha
    alp = mask.astype(float)

    foreground = cv2.multiply(alp, fg)
    background = cv2.multiply(1.0 - alp, bg)
    out_image = cv2.add(foreground, background)

    return out_image

def polygon_mask(image: list, landmark_object: dict, point_id_list: list):
    """
    Creates a alpha mask of the desired polygons
    Image is the return value from the cv2.imread function
    Point_id is the point id of the polygon vertices
    """
    # Initialize Variables
    blank = np.zeros(image.shape, float) 
    
    for id_list in point_id_list: 
        temp = []
        for Id in id_list:
            temp.append(LandPoint.get_point_by_id(landmark_object, Id))

        vertices = np.array(temp)
        alpha_mask = cv2.fillConvexPoly(blank, vertices, (1,1,1))

    return alpha_mask

def mask(foreground, background, alpha): 
    """
    Simply Creates the alpha mask Image
    """
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    out_image = cv2.add(foreground, background)

    return out_image
