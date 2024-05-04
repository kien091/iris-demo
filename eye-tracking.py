import mediapipe as mp
import cv2

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # 468 points
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_landmark(image):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )
    image.flags.writeable = False
    results = face_mesh.process(image)
    landmarks = results.multi_face_landmarks[0].landmark
    return results, landmarks

def draw_landmark(image, results):
    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = image,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_list = face_landmarks, 
                landmark_drawing_spec = None, 
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # mp_drawing.draw_landmarks(
            #     image = image,
            #     landmark_list = face_landmarks, 
            #     connections = mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec = None, 
            #     connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
            # )
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks, 
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None, 
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    return image

path_image = "data/iris.jpg"
img = cv2.imread(path_image)
annotated_img = img.copy()

result, landmarks = get_landmark(image = img)

annotated_img = draw_landmark(image = annotated_img, results = result)

cv2.imshow("Image", img)
cv2.imshow("Annotated Image", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()