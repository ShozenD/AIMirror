import matplotlib.pyplot as plt
import face_recognition
import face_point_extraction

my_face = face_recognition.load_image_file("../photo/aligned_face_0.jpg")
other_face = face_recognition.load_image_file("../photo/aligned_face_1.jpg")

my_face_flp = face_recognition.face_landmarks(my_face)
other_face_flp = face_recognition.face_landmarks(other_face)

id_list = [(9, 64), (52, 34), (15, 31), (28, 43), (25, 45), (46, 17), (28, 31), (32, 36), (31, 43)]

my_coords = face_point_extraction.get_points_by_ID(my_face_flp, id_list)
other_coords = face_point_extraction.get_points_by_ID(other_face_flp, id_list)

my_dist = face_point_extraction.get_distance(my_coords)
other_dist = face_point_extraction.get_distance(other_coords)

my_mid = face_point_extraction.get_midpoint(my_coords)
other_mid = face_point_extraction.get_midpoint(my_coords)

plt.figure(figsize=(15,10))
plt.subplot(121),plt.imshow(my_face)
for ele in my_coords:
    plt.plot(ele[0], ele[1])

for dist, mid in zip(my_dist, my_mid):
    x, y = mid
    plt.text(x, y, str(dist))

plt.subplot(122),plt.imshow(other_face)
for ele in other_coords:
    plt.plot(ele[0], ele[1])

for dist, mid in zip(other_dist, other_mid):
    x, y = mid
    plt.text(x, y, str(dist))

plt.show()
