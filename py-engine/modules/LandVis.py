"""
This Module is Used to Visualize the Facial Landmark Points
"""
import cv2
import modules.LandPoint as LandPoint
import numpy as np

def visualize_flp(image: list, landmark_object: list):
    visualized = image
    for face in landmark_object:
        for part in face:
            for point in face[part]:
                visualized = cv2.circle(visualized, point, radius = 1, color=(255,100,100), thickness = 2)
    
    return visualized 

def visualize_polygons(image: list, landmark_object: list, point_id_list: list, line_color = (150,100,255)):
    """
    Creates a image in which the desired polygon is surrounded by line segments.
    """
    # Initialize Variables
    visualized = image

    for id_list in point_id_list: 
        temp = []    
        # Create Coodinate Points 
        for point in id_list:
            temp.append(LandPoint.get_point_by_id(landmark_object, point))

        # Make Compatible with cv2
        pts = np.array(temp, np.int32)
        pts = pts.reshape((-1,1,2))

        visualized = cv2.polylines(visualized, [pts], isClosed=True, color=line_color, thickness=1, lineType=4)

    return visualized