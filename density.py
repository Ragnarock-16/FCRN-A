import os
import glob
import numpy as np
import cv2

# Function to generate Gaussian value
def gaussian(x, y, center_x, center_y, sigma):
    """
    Generate a Gaussian value at position (x, y) given the center (center_x, center_y) and sigma.
    """
    return np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

# Function to generate density surface (heatmap) from YOLO coordinates
def generate_density_surface(image_size, yolo_coords, sigma=10, gauss_amplitude=10000):
    """
    Generate a density surface (heatmap) from YOLO coordinates.
    
    Parameters:
    - image_size: tuple (height, width) for the size of the output heatmap.
    - yolo_coords: list of YOLO coordinates in the format [x_center, y_center, width, height].
    - sigma: standard deviation of the Gaussian function, controls the spread of the peak.
    - gauss_amplitude: multiplier for the Gaussian values to ensure visible intensity.
    
    Returns:
    - density_surface: Generated heatmap of the same size as the image.
    """
    # Initialize a blank image (density surface) with zeros (black background)
    density_surface = np.zeros(image_size)

    # Iterate through each object in the YOLO coordinates
    for coord in yolo_coords:
        # Convert YOLO normalized coordinates to pixel coordinates
        x_center, y_center, width, height = coord
        
        # Convert to pixel coordinates (assuming image width and height are 1.0-1.0 normalized)
        pixel_x = int(x_center * image_size[1])
        pixel_y = int(y_center * image_size[0])

        # Check if the center is within image bounds
        if pixel_x >= 0 and pixel_x < image_size[1] and pixel_y >= 0 and pixel_y < image_size[0]:
            # Apply a Gaussian at the center of the object
            for y in range(max(0, pixel_y - 2 * sigma), min(image_size[0], pixel_y + 2 * sigma)):
                for x in range(max(0, pixel_x - 2 * sigma), min(image_size[1], pixel_x + 2 * sigma)):
                    # Compute Gaussian value at (x, y) and multiply by the amplitude
                    density_surface[y, x] += gauss_amplitude * gaussian(x, y, pixel_x, pixel_y, sigma)
        else:
            print(f"Skipping Gaussian for out-of-bound coordinate: ({pixel_x}, {pixel_y})")
    
    # Normalize the density surface to the range [0, 255]
    density_surface = np.clip(density_surface, 0, None)  # Clip to avoid overflow
    
    # Scale the density surface to 0-255 range
    if np.max(density_surface) > 0:
        density_surface = (density_surface / np.max(density_surface)) * 255
    
    # Convert to uint8 for visualization
    density_surface = density_surface.astype(np.uint8)
    
    return density_surface

# Function to parse YOLO label file
def parse_yolo_labels(label_file, image_width, image_height):
    """
    Parse YOLO label file and convert normalized coordinates to pixel coordinates.
    
    Parameters:
    - label_file: Path to the YOLO label file.
    - image_width: Width of the image.
    - image_height: Height of the image.
    
    Returns:
    - yolo_coords: List of parsed YOLO coordinates as [x_center, y_center, width, height].
    """
    yolo_coords = []
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        
        # YOLO format: class_id x_center y_center width height (all normalized)
        x_center = float(parts[1])  # Normalized center x
        y_center = float(parts[2])  # Normalized center y
        width = float(parts[3])  # Normalized width
        height = float(parts[4])  # Normalized height
        
        yolo_coords.append([x_center, y_center, width, height])
    
    return yolo_coords

# Function to process and save heatmap for each image in a folder
def process_and_save_heatmaps(image_folder, label_folder, output_folder, sigma=2, gauss_amplitude=1):
    """
    Process all images in the given folder, generate heatmaps, and save them to the output folder.
    
    Parameters:
    - image_folder: Folder containing input images.
    - label_folder: Folder containing the corresponding YOLO label files.
    - output_folder: Folder where heatmaps will be saved.
    - sigma: Gaussian spread (default is 2).
    - gauss_amplitude: Gaussian intensity (default is 1).
    """
    # Get all image file paths in the image folder
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # Assuming images are .jpg

    # Process each image
    for image_path in image_paths:
        # Get the base name of the image (without the extension)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Construct the corresponding label file path
        label_file_path = os.path.join(label_folder, f"{image_name}.txt")
        
        # Check if the label file exists
        if not os.path.exists(label_file_path):
            print(f"Label file for {image_name} does not exist, skipping...")
            continue

        # Load the image to get its dimensions
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # Parse YOLO label file to get object coordinates
        yolo_coords = parse_yolo_labels(label_file_path, image_width, image_height)

        # Generate the density surface (heatmap) for the image
        density_surface = generate_density_surface((image_height, image_width), yolo_coords, sigma=sigma, gauss_amplitude=gauss_amplitude)

        # Apply a color map (Jet or any other) to visualize the density
        colored_heatmap = cv2.applyColorMap(density_surface, cv2.COLORMAP_INFERNO)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the heatmap with the original image name (no "_heatmap" suffix)
        output_file_path = os.path.join(output_folder, f"{image_name}.jpg")
        cv2.imwrite(output_file_path, colored_heatmap)

        print(f"Heatmap for {image_name} saved to {output_file_path}")

# Example usage
image_folder = "custom/images/train/"  # Folder containing images
label_folder = "custom/labels/train/"  # Folder containing YOLO label files
output_folder = "output/heatmaps"  # Folder where heatmaps will be saved

# Process all images and save the heatmaps
process_and_save_heatmaps(image_folder, label_folder, output_folder, sigma=6, gauss_amplitude=10)
