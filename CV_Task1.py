import cv2
import numpy as np
import cv2.aruco as aruco

import cv2
import os

# Der Pfad des aktuellen Skripts
script_dir = os.path.dirname(__file__)

# Der relative Pfad zum Bild im 'img'-Ordner
image__folder_path = os.path.join(script_dir, 'Img')
output_folder_path = os.path.join(script_dir, 'Output')
evaluation_folder_path = os.path.join(script_dir, 'Evaluation')
img_list = [datei for datei in os.listdir(image__folder_path) if datei.endswith('.jpg')]
poster_path = os.path.join(script_dir, 'poster.jpg')


### function to find slope 
def slope(p1,p2):
    x1,y1=p1
    x2,y2=p2
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'
### main function to draw lines between two points
def drawLine(image,p1,p2):
    x1,y1=p1
    x2,y2=p2
    ### finding slope
    m=slope(p1,p2)
    ### getting image shape
    h,w=image.shape[:2]

    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)
    return image


    #show the image after each step
def show_image(image, Function):
    cv2.namedWindow(Function, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(Function, image)

def extract_lines_from_edges(edges):
    """Extracts lines from the detected edges using the Hough Transform.

    Args:
        edges: The image with detected edges.

    Returns:
        A list of lines, where each line is a tuple of four numbers representing the coordinates of the two endpoints.
    """

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10,)
    return lines

def detect_edges(image):
    """Detects edges in an image using Canny edge detection.

    Args:
        image_path: The path to the image file.

    Returns:
        The image with detected edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

for img in img_list:
    try:
        image_path = os.path.join(image__folder_path,img)
        # Load the image containing the ArUco markers
        image = cv2.imread(image_path)
        # Load the poster image
        poster = cv2.imread(poster_path)

        #define constants 
        poster_scale=13 #define the size of the Poster in the final Image (1 is the size of the marker)
        y_offset = 130

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
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        parameters = aruco.DetectorParameters()
        arucocorners, ids, Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

        M = cv2.getPerspectiveTransform(scaled_poster_corners, arucocorners[0][0])
        transformed_poster_corners = cv2.perspectiveTransform(poster_corners.reshape(-1, 1, 2), M)
        pts = transformed_poster_corners.reshape(4, 2).astype(np.int32)

        transformed_poster= cv2.warpPerspective(poster, M,(image.shape[1], image.shape[0]))
        #show_image(transformed_poster,"Transformed Poster")

        # Create mask with zeros
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

        # Invert mask to keep everything outside the polygon area
        mask = cv2.bitwise_not(mask)
        #show_image(mask,"Mask")

        # Apply mask to image to remove the polygon area
        masked_image = cv2.bitwise_and(image, mask)
        #show_image(masked_image, "Masked image")
        final_image = cv2.bitwise_or(masked_image, transformed_poster)

        cv2.imwrite(os.path.join(output_folder_path,img),final_image)

        edges = detect_edges(final_image)
        edge_lines = extract_lines_from_edges(edges)

        for line in edge_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        
        # final_image = cv2.polylines(final_image, [pts], isClosed=True, color=(255, 255, 0), thickness=outline_thickness)
        final_image = drawLine(final_image, pts[0],pts[1])
        final_image = drawLine(final_image, pts[0],pts[3])
        final_image = drawLine(final_image, pts[1],pts[2])
        final_image = drawLine(final_image, pts[2],pts[3])



        #show_image(final_image, "final image")
        cv2.imwrite(os.path.join(evaluation_folder_path,img),final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
