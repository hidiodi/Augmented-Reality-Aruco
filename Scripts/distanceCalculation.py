import cv2
import numpy as np

def calculate_distance_to_bbox(image_height, bbox, camera_height, focal_length=None):
    """
    Calculate the distance from the camera to the bounding box.

    Parameters:
    - image: The input image.
    - bbox: The bounding box coordinates (x, y, width, height).
    - camera_height: The height of the camera from the ground.
    - focal_length: The focal length of the camera (optional).

    Returns:
    - distance: The distance from the camera to the bounding box.
    """
    # Extract the bounding box dimensions
    x, y, width, height = bbox

    # Calculate the perceived height of the object in the image
    perceived_height = height

    # Calculate the real height of the object (assuming the object is on the ground)
    real_height = camera_height

    # If focal length is not provided, estimate it using known parameters
    if focal_length is None:
        # Assuming a standard sensor size and field of view
        sensor_height = 24  # in mm (for a full-frame sensor)
        fov = 60  # field of view in degrees

        # Calculate the focal length using the sensor size and field of view
        focal_length = (image_height * sensor_height) / (2 * np.tan(np.deg2rad(fov / 2)))

    # Calculate the distance using the pinhole camera model
    distance = (real_height * focal_length) / perceived_height

    return distance

# Example usage
if __name__ == "__main__":
    # Load an example image
    image = cv2.imread('example.jpg')

    # Define the bounding box coordinates (x, y, width, height)
    bbox = (100, 200, 50, 100)

    # Define the camera height and focal length
    camera_height = 1.5  # in meters
    focal_length = 800  # in pixels

    # Calculate the distance
    distance = calculate_distance_to_bbox(image, bbox, camera_height, focal_length)
    print(f"Distance to bounding box: {distance} meters")