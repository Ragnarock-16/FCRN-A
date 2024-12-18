import os
from PIL import Image, ImageDraw

def load_bounding_boxes(file_path):
    """
    Reads bounding boxes from a file.

    Args:
        file_path (str): Path to the file containing bounding boxes in YOLO format.

    Returns:
        list of tuples: Bounding boxes as (class_id, x_center, y_center, width, height).
    """
    bboxes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes

def draw_bounding_boxes(image_path, bboxes, output_path=None):
    """
    Draws bounding boxes on an image.

    Args:
        image_path (str): Path to the input image.
        bboxes (list of tuples): Bounding boxes in YOLO format (class_id, x_center, y_center, width, height).
        output_path (str, optional): Path to save the output image. If None, the image is not saved.

    Returns:
        Image object: The image with bounding boxes drawn.
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for bbox in bboxes:
        class_id, x_center, y_center, box_width, box_height = bbox

        # Convert normalized coordinates to absolute pixel values
        x_center_abs = x_center * width
        y_center_abs = y_center * height
        box_width_abs = box_width * width
        box_height_abs = box_height * height

        # Calculate the top-left and bottom-right corners of the bounding box
        x_min = x_center_abs - (box_width_abs / 2)
        y_min = y_center_abs - (box_height_abs / 2)
        x_max = x_center_abs + (box_width_abs / 2)
        y_max = y_center_abs + (box_height_abs / 2)

        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Optionally, add the class ID label near the top-left corner of the box
        draw.text((x_min, y_min - 10), str(class_id), fill="red")

    # Save the output image if an output path is specified
    if output_path:
        image.save(output_path)

    return image

def draw_bounding_boxes_from_file(image_path, bbox_file, output_path=None):
    """
    Draws bounding boxes on an image using data from a bounding box file.

    Args:
        image_path (str): Path to the input image.
        bbox_file (str): Path to the file containing bounding boxes in YOLO format.
        output_path (str, optional): Path to save the output image. If None, the image is not saved.

    Returns:
        Image object: The image with bounding boxes drawn.
    """
    # Load bounding boxes from the file
    bboxes = load_bounding_boxes(bbox_file)
    return draw_bounding_boxes(image_path, bboxes, output_path)

# Example usage
if __name__ == "__main__":
    # Paths to the image and bounding box file
    image_path = "custom/images/train/BM_GRAZ_HE_0001_01_0.jpg"
    bbox_file = "/Users/nour/Documents/M2/imagerie_biomedical/FCRN/custom/labels/train/BM_GRAZ_HE_0001_01_0.txt"  # File containing bounding box data

    # Draw bounding boxes and save the output image
    output_image = draw_bounding_boxes_from_file(image_path, bbox_file, "output_image.jpg")
    output_image.show()

    # Batch processing example
    images_dir = "images"
    labels_dir = "labels"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(images_dir):
        if image_file.endswith(".jpg"):
            base_name = os.path.splitext(image_file)[0]
            bbox_file = os.path.join(labels_dir, f"{base_name}.txt")
            image_path = os.path.join(images_dir, image_file)
            output_path = os.path.join(output_dir, f"{base_name}_with_boxes.jpg")

            if os.path.exists(bbox_file):
                draw_bounding_boxes_from_file(image_path, bbox_file, output_path)
