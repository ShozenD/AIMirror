from PIL import Image, ImageDraw
import face_recognition
import sys

PATH  = sys.argv[1]

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(PATH)

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

def apply_makeup(image, landmark_oject, eyeline_color=(0,0,0,90), lip_color=(240,55,55,128)):
    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=eyeline_color)
        d.polygon(face_landmarks['right_eyebrow'], fill=eyeline_color)
        d.line(face_landmarks['left_eyebrow'], fill=eyeline_color, width=2)
        d.line(face_landmarks['right_eyebrow'], fill=eyeline_color, width=2)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=lip_color)
        d.polygon(face_landmarks['bottom_lip'], fill=lip_color)
        #d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=2)
        #d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        #d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        #d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=eyeline_color, width=3)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=eyeline_color, width=3)

    pil_image.save('../photo/virtual_make.jpg')

apply_makeup(image, face_landmarks_list)
