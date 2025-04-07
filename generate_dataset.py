from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ulid import ULID
from tqdm import tqdm
import argparse
import shutil
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# bar_format = "[{l_bar}{bar} {rate_fmt}{postfix} | {n_fmt}/{total_fmt} {elapsed}<{remaining}]"
bar_format = "{l_bar}{bar}|"

# List of fonts to use
fonts = [
    # { 'path': 'fonts/cour.ttf', 'size': 16, 'height_control': 8 },
    { 'path': 'fonts/T_win10.otf', 'size': 5, 'left_padding': 1, 'top_padding': 10, 'right_padding': 1, 'bottom_padding': 0 },
    { 'path': 'fonts/T_win15.otf', 'size': 8, 'left_padding': 2, 'top_padding': 3, 'right_padding': 0, 'bottom_padding': 2 },
    { 'path': 'fonts/T_win80.otf', 'size': 8, 'left_padding': 1, 'top_padding': -16, 'right_padding': 0, 'bottom_padding': 20 },
    # { 'path': 'fonts/Xfont80.otf', 'size': 14, 'height_control': 32 },
    # { 'path': 'fonts/Xfontlg.otf', 'size': 12, 'height_control': -24 },
    # { 'path': 'fonts/Xfontsm.otf', 'size': 12, 'height_control': 24 },
]

def generate_random_ulid():
    """Generate a random ULID string."""

    ulid = ULID()
    ulid_str = str(ulid)

    return (ulid_str, ulid_str)

def generate_random_date_string():
    """Generate a random date string in the format MM/DD/YYYY-NNN."""

    ulid = ULID()
    mm = str(random.randint(10, 99)).zfill(2)
    dd = str(random.randint(10, 99)).zfill(2)
    yyyy = str(random.randint(1000, 9999)).zfill(4)
    nnn = str(random.randint(100, 999)).zfill(3)

    return (str(ulid), f"{mm}/{dd}/{yyyy}-{nnn}")

def generate_randon_number_string():
    """Generate a random number string in the format NNNNNNNN."""
    
    ulid = ULID()
    nnnnnnnn = str(random.randint(10000000, 99999999)).zfill(8)

    return (str(ulid), f"{nnnnnnnn}")

def generate_random_string(length=2):
    """Generate a random string of uppercase letters and digits."""

    ulid = ULID()
    characters = '0123456789'
    string = ''.join(random.choice(characters) for _ in range(length))

    return (str(ulid), f"{string}")

# List of random types to generate
rand_types = [
    generate_random_ulid,
    generate_random_date_string,
    generate_randon_number_string,
    # generate_random_string,
]

# Output directory for generated images
output_dir = "dataset"

def recreate_output_folder(output_dir):
    """Delete and recreate the output directory."""

    if os.path.exists(output_dir):
        print(f"Deleting {output_dir} folder...")
        shutil.rmtree(output_dir)
    print(f"Creating {output_dir} folder...")
    os.makedirs(output_dir, exist_ok=True)

def generate_image(dataset_id, text, font_path, font_size, paddings, output_dir):
    """Generate an image with random text using the specified font."""

    try:
        # Load the font
        font = ImageFont.truetype(font_path, size=font_size)
        left_padding, top_padding, right_padding, bottom_padding = paddings

        # Set the image size
        width, height = 320, 100
        
        # Create a random noise image
        noise = np.random.randint(195, 255, (height, width), dtype=np.uint8)
        noise_image = Image.fromarray(noise, mode='L').filter(ImageFilter.GaussianBlur(radius=1))

        # Create an RGB image and paste the noise background
        image = Image.new("RGB", (width, height), "white")
        image.paste(noise_image, (0, 0))

        # Draw the text
        draw = ImageDraw.Draw(image)

        # Starting position (adjust as needed)
        x, y = 22, 26  # Baseline coordinates for text

        # Track character positions for .box file
        box_entries = []
        x_offset = x  # Start at initial x

        # Get font metrics
        ascent, descent = font.getmetrics()

        for char in text:
            # Get character width and bounding box
            char_width = font.getlength(char)
            left = x_offset + left_padding # Left padding
            right = x_offset + char_width - right_padding # Right padding

            # Tesseract uses bottom-left origin, so invert y-coordinates
            y_top = y - ascent - top_padding  # Top of the character (PIL y decreases upward)
            y_bottom = y + descent + bottom_padding  # Bottom of the character

            # Convert PIL coordinates (top-left origin) to Tesseract (bottom-left)
            tesseract_top = height - y_top
            tesseract_bottom = height - y_bottom

            # Ensure coordinates are within image bounds
            left = max(0, min(left, width))
            right = max(0, min(right, width))
            tesseract_top = max(0, min(tesseract_top, height))
            tesseract_bottom = max(0, min(tesseract_bottom, height))

            # Add to .box entries
            box_entries.append(f"{char} {int(left)} {int(tesseract_bottom)} {int(right)} {int(tesseract_top)} 0")

            x_offset += char_width  # Move to next character position

        # Draw the text (using original method for rendering)
        draw.text((x, y), text, font=font, fill="black")

        # Draw bounding boxes for debugging (optional)
        for entry in box_entries:
            char, left, bottom, right, top, _ = entry.split()
            left, bottom, right, top = map(int, (left, bottom, right, top))
            draw.rectangle([left, height - top, right, height - bottom], outline="red", width=1)            

        # Save as TIFF (required for Tesseract training)
        image.save(os.path.join(output_dir, f"{dataset_id}.tif"))
        # image.save(os.path.join(output_dir, f"{dataset_id}.png"))

        # Save .box file (same name as image)
        with open(os.path.join(output_dir, f"{dataset_id}.box"), "w") as f:
            f.write("\n".join(box_entries))

        # Save ground truth text
        with open(os.path.join(output_dir, f"{dataset_id}.gt.txt"), "w") as f:
            f.write(text)


        ### old code

        # # Use textbbox to calculate text size
        # text_bbox = draw.textbbox((0, 0), text, font=font)
        # text_width = text_bbox[2] - text_bbox[0]
        # text_height = text_bbox[3] - text_bbox[1]
        # # x = (width - text_width) // 2
        # # y = (height - text_height - height_control) // 2
        # x = 22
        # y = 26
        # draw.text((x, y), text, font=font, fill="black")

        # # new a new rectangle with the x and y coordinates
        # # (left, top, right, bottom) bounding box
        # # bounding box (x, y, with, height)
        # text_bbox = (x, y - text_height, x + text_width, y + text_height)
        # # Draw a rectangle around the text
        # # draw.rectangle((24, 16, 8, 15), outline="red", width=1)
        
        # # draw.rectangle(text_bbox, outline="red", width=1)
        # # draw.rectangle(((x, y), (x + text_width, y + text_height)), outline='Red')

        # # Save the image
        # image.save(os.path.join(output_dir, f"{dataset_id}.png"))
        # # Save the ground truth text
        # with open(os.path.join(output_dir, f"{dataset_id}.gt.txt"), "w") as f:
        #     f.write(text)
        
    except Exception as e:
        tqdm.write(f"Error with font {font_path}: {e}")

def generate_images_for_font(font, qty, output_dir, progress_bar):
    """Generate images for a specific font."""
    font_path = font['path']
    font_size = font['size']
    left_padding = font['left_padding']
    top_padding = font['top_padding']
    right_padding = font['right_padding']
    bottom_padding = font['bottom_padding']
    paddings = (left_padding, top_padding, right_padding, bottom_padding)

    for rand_type in rand_types:
        for _ in range(qty):
            dataset_id, text = rand_type()
            generate_image(dataset_id, text, font_path, font_size, paddings, output_dir)

            progress_bar.update(1)  # Update the shared progress bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", required=False, type=int, help="Quantity of images to generate for each font")
    parser.add_argument("-o", required=False, type=str, help="Output directory for generated images", default=output_dir)
    parser.add_argument("-d", required=False, help="Delete the output folder before generating images", action="store_true")
    args = parser.parse_args()

    qty = args.q
    if qty and qty < 1:
        raise ValueError("Quantity must be greater than 0")
    if args.o:
        output_dir = args.o
    if args.d:
        recreate_output_folder(args.o)

    if qty:
        # Calculate the total number of tasks for the progress bar
        total_tasks = len(fonts) * len(rand_types) * qty

        # Create a shared tqdm progress bar
        with tqdm(total=total_tasks, desc="Generating Images", bar_format=bar_format) as progress_bar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(generate_images_for_font, font, qty, output_dir, progress_bar)
                    for font in fonts
                ]

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    future.result()
