from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ulid import ULID
from tqdm import tqdm
import argparse
import shutil
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# bar_format = "[{l_bar}{bar} {rate_fmt}{postfix} | {n_fmt}/{total_fmt} {elapsed}<{remaining}]"
bar_format = "{l_bar}{bar}|"

# List of fonts to use
fonts = [
    { 'path': 'fonts/T_win10.otf', 'size': 5, 'left_padding': 1, 'top_padding': 10, 'right_padding': 1, 'bottom_padding': 0 },
    { 'path': 'fonts/T_win15.otf', 'size': 8, 'left_padding': 2, 'top_padding': 3, 'right_padding': 0, 'bottom_padding': 4 },
    { 'path': 'fonts/T_win80.otf', 'size': 8, 'left_padding': 1, 'top_padding': -16, 'right_padding': 0, 'bottom_padding': 20 },
    { 'path': 'fonts/Xfont80.otf', 'size': 14, 'left_padding': 1, 'top_padding': -30, 'right_padding': 0, 'bottom_padding': 28 },
    { 'path': 'fonts/Xfontlg.otf', 'size': 12, 'left_padding': 2, 'top_padding': -1, 'right_padding': 0, 'bottom_padding': 4 },
    { 'path': 'fonts/Xfontsm.otf', 'size': 12, 'left_padding': 1, 'top_padding': -25, 'right_padding': 1, 'bottom_padding': 25 },
]

# Output directory for generated images
output_dir = "tesstrain/data/Meditech-ground-truth"

# Add a global counter and lock
image_counter = 0
counter_lock = threading.Lock()

def get_next_image_id():
    """Get the next image ID in a thread-safe way."""
    global image_counter
    with counter_lock:
        current_id = image_counter
        image_counter += 1
    return current_id

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

def recreate_output_folder(output_dir):
    """Delete and recreate the output directory."""

    if os.path.exists(output_dir):
        print(f"Deleting {output_dir} folder...")
        shutil.rmtree(output_dir)
    print(f"Creating {output_dir} folder...")
    os.makedirs(output_dir, exist_ok=True)

def generate_image(image_id, text, font_path, font_size, paddings, output_dir, debug):
    """Generate an image with random text using the specified font."""

    # Set the random seed for reproducibility
    left_padding, top_padding, right_padding, bottom_padding = paddings

    try:
        # Load the font
        font = ImageFont.truetype(font_path, size=font_size, encoding="unic")

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

        if debug:
            # Draw bounding boxes for debugging (optional)
            for entry in box_entries:
                char, left, bottom, right, top, _ = entry.split()
                left, bottom, right, top = map(int, (left, bottom, right, top))
                draw.rectangle([left, height - top, right, height - bottom], outline="red", width=1)            

        # Save as TIFF (required for Tesseract training)
        image.save(os.path.join(output_dir, f"eng_{image_id}.tif"))
        # image.save(os.path.join(output_dir, f"{image_id}.png"))

        # Save .box file (same name as image)
        with open(os.path.join(output_dir, f"eng_{image_id}.box"), "w") as f:
            f.write("\n".join(box_entries))

        # Save ground truth text
        with open(os.path.join(output_dir, f"eng_{image_id}.gt.txt"), "w") as f:
            f.write(text)
        
    except Exception as e:
        tqdm.write(f"Error with font {font_path}: {e}")

def generate_images_for_font(font, qty, output_dir, progress_bar, debug):
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
            _, text = rand_type()
            image_id = get_next_image_id()  # Get sequential ID instead
            generate_image(image_id, text, font_path, font_size, paddings, output_dir, debug)
            progress_bar.update(1)  # Update the shared progress bar

# List of random types to generate
rand_types = [
    generate_random_ulid,
    generate_random_date_string,
    generate_randon_number_string,
    # generate_random_string,
]

if __name__ == "__main__":
    # Reset counter before generating images
    image_counter = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", required=False, type=int, help="Quantity of images to generate for each font")
    parser.add_argument("-o", required=False, type=str, help="Output directory for generated images", default=output_dir)
    parser.add_argument("-d", required=False, help="Delete the output folder before generating images", action="store_true")
    parser.add_argument("-debug", required=False, help="Enable debug mode", action="store_true")
    args = parser.parse_args()

    qty = args.q
    if qty and qty < 1:
        raise ValueError("Quantity must be greater than 0")
    if args.o:
        output_dir = args.o
    if args.d:
        recreate_output_folder(args.o)

    debug = args.debug or False

    if qty:
        # Calculate the total number of tasks for the progress bar
        total_tasks = len(fonts) * len(rand_types) * qty

        # Create a shared tqdm progress bar
        with tqdm(total=total_tasks, desc="Generating Images", bar_format=bar_format) as progress_bar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(generate_images_for_font, font, qty, output_dir, progress_bar, debug)
                    for font in fonts
                ]

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    future.result()
