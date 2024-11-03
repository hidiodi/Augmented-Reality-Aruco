import cv2
import numpy as np
import cv2.aruco as aruco

import cv2
import os

# Der Pfad des aktuellen Skripts
script_dir = os.path.dirname(__file__)

# Der relative Pfad zum Bild im 'img'-Ordner
image__folder_path = os.path.join(script_dir, 'img')
output_folder_path = os.path.join(script_dir, 'output')
img_list = [datei for datei in os.listdir(image__folder_path) if datei.endswith('.jpg')]
poster_path = os.path.join(script_dir, 'poster.jpg')

for img in img_list:
    image_path = os.path.join(image__folder_path,img)
    # Load the image containing the ArUco markers
    image = cv2.imread(image_path)
    # Load the poster image
    poster = cv2.imread(poster_path)

    #define constants 
    poster_scale=14 #define the size of the Poster in the final Image (1 is the size of the marker)
    y_offset = 80

    #show the image after each step
    def show_image(image, Function):
        cv2.namedWindow(Function, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(Function, image)


    poster_height, poster_width = poster.shape[:2]

    poster_corners = np.array([
        [0, 0],                   # Top-left
        [poster_width, 0],       # Top-right
        [poster_width, poster_height],  # Bottom-right
        [0, poster_height]        # Bottom-left
    ], dtype=np.float32)

    [x, y] = np.mean(poster_corners, axis=0)
    scaled_width = poster_width / poster_scale
    scaled_height = poster_height / poster_scale

    scaled_poster_corners = np.array([
        [x - scaled_width, y - scaled_height + y_offset],  # Top-left
        [x + scaled_width, y - scaled_height + y_offset],  # Top-right
        [x + scaled_width, y + scaled_height + y_offset],  # Bottom-right
        [x - scaled_width, y + scaled_height + y_offset]   # Bottom-left
    ], dtype=np.float32)

    # Detect ArUco markers in the image
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    arucocorners, ids, Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    M = cv2.getPerspectiveTransform(scaled_poster_corners, arucocorners[0][0])
    transformed_poster_corners = cv2.perspectiveTransform(poster_corners.reshape(-1, 1, 2), M)
    pts = transformed_poster_corners.reshape(4, 2).astype(np.int32)

    transformed_poster= cv2.warpPerspective(poster, M,(image.shape[1], image.shape[0]))
    show_image(transformed_poster,"Transformed Poster")

    # Create mask with zeros
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Invert mask to keep everything outside the polygon area
    mask = cv2.bitwise_not(mask)
    show_image(mask,"Mask")

    # Apply mask to image to remove the polygon area
    masked_image = cv2.bitwise_and(image, mask)
    show_image(masked_image, "Masked image")
    final_image = cv2.bitwise_or(masked_image, transformed_poster)

    outline_thickness = 12  # Adjust as needed
    final_image = cv2.polylines(final_image, [pts], isClosed=True, color=(255, 255, 0), thickness=outline_thickness)

    #show_image(final_image, "final image")
    cv2.imwrite(os.path.join(output_folder_path,img),final_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
