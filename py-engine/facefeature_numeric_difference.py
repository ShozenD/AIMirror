from PIL import Image, ImageDraw
import sys
import face_recognition

# Load image from command line
input_1 = sys.argv[1]
input_2 = sys.argv[2]

user_face = face_recognition.load_image_file(input_1)
# Load image to compare against
other_face = face_recognition.load_image_file(input_2)

# Find the facial features in both images
# There is only supposed to be one face in each photo
user_face_landmark = face_recognition.face_landmarks(user_face)
other_face_landmark = face_recognition.face_landmarks(other_face)

# Get the facial encodings for both images
user_face_encoding = face_recognition.face_encodings(user_face)[0]
other_face_encoding =  face_recognition.face_encodings(other_face)[0]

# List of known encodings(In this case only one)
known_encodings = [
    user_face_encoding
]

# Calculate Euclidean Face Distance
face_distances = face_recognition.face_distance(known_encodings, other_face_encoding)

for user_landmark, other_landmark in zip(user_face_landmark, other_face_landmark):
    # lets print out the points!
    #for user_feature in user_landmark.keys():
        #print("The {} in user's face has the following points: {}".format(user_feature, user_landmark[user_feature]))

    #for other_feature in other_landmark.keys():
        #print("The {} in other's face has the following points: {}".format(other_feature, other_landmark[other_feature]))

    # Let's trace out each facial feature in the image and overlay it with the image being compared
    pil_image = Image.fromarray(user_face)
    d = ImageDraw.Draw(pil_image)

    for user_feature, other_feature in zip(user_landmark.keys(), other_landmark.keys()):
        d.line(user_landmark[user_feature], width = 3)
        d.line(other_landmark[other_feature], width = 3, fill="#ff0000")

    pil_image.show()

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
