import cv2
import numpy as np
import cv2.aruco as aruco

# Load the image containing the ArUco markers
image = cv2.imread(r'd:\VisualStudio\Computer Vision Task 1\20221115_113319.jpg')
# Load the poster image
poster = cv2.imread(r'd:\VisualStudio\Computer Vision Task 1\poster.jpg')

#define constants 
poster_scale=8 #define the size of the Poster in the final Image (1 is the size of the marker)

#show the image after each step
def show_image(image, Function):
    cv2.namedWindow(Function, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(Function, image)


poster_height, poster_width = poster.shape[:2]

poster_corners = np.array([
    [0, 0],                                 # Top-left
    [poster_width, 0],                      # Top-right
    [poster_width, poster_height],          # Bottom-right
    [0, poster_height]                      # Bottom-left
], dtype=np.float32)

[x, y] = np.mean(poster_corners, axis=0)
scaled_width = poster_width / poster_scale
scaled_height = poster_height / poster_scale

scaled_poster_corners = np.array([
    [x - scaled_width, y - scaled_height],  # Top-left
    [x + scaled_width, y - scaled_height],  # Top-right
    [x + scaled_width, y + scaled_height],  # Bottom-right
    [x - scaled_width, y + scaled_height]   # Bottom-left
], dtype=np.float32)

# Detect ArUco markers in the image
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

arucocorners, ids, Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
# Get Perspective Transform
M = cv2.getPerspectiveTransform(scaled_poster_corners, arucocorners[0][0])
transformed_poster_corners = cv2.perspectiveTransform(poster_corners.reshape(-1, 1, 2), M)
pts = transformed_poster_corners.reshape(4, 2).astype(np.int32)

# Apply Perspective Transform to the Poster
scaled_poster = cv2.resize(poster, (poster_width // poster_scale, poster_height //poster_scale))
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

outline_thickness = 0  # Adjust as needed
final_image = cv2.polylines(final_image, [pts], isClosed=True, color=(255, 255, 0), thickness=outline_thickness)

show_image(final_image, "final image")
cv2.imwrite("finalimage.jpg",final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
