import cv2
import dlib
import openface
import sys
file = sys.argv[1]

predictor_model = "/Users/shozendan/Documents/AIMirror/data/shape_predictor_68_face_landmarks.dat"

image = cv2.imread(file)

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file))

for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # Use openface to calculate and perform the face alignment
    alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    # Save the aligned image to a file
    cv2.imwrite("/Users/shozendan/Documents/AIMirror/photo/aligned_face_{}.jpg".format(i), alignedFace)
