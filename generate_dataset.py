from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ulid import ULID
from tqdm import tqdm
import argparse
import shutil
import numpy
import os
import random

bar_format = "[{l_bar}{bar} {rate_fmt}{postfix} | {n_fmt}/{total_fmt} {elapsed}<{remaining}]"

# List of fonts to use
fonts = [
    { 'path': 'fonts/T_win10.otf', 'size': 5, 'height_control': -32 },
    { 'path': 'fonts/T_win15.otf', 'size': 8, 'height_control': -22 },
    { 'path': 'fonts/T_win80.otf', 'size': 8, 'height_control': 16 },
    { 'path': 'fonts/Xfont80.otf', 'size': 14, 'height_control': 32 },
    { 'path': 'fonts/Xfontlg.otf', 'size': 12, 'height_control': -24 },
    { 'path': 'fonts/Xfontsm.otf', 'size': 12, 'height_control': 24 },
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

# List of random types to generate
rand_types = [
    generate_random_ulid,
    generate_random_date_string,
    generate_randon_number_string
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

def generate_image(dataset_id, text, font_path, font_size, height_control, output_dir):
    """Generate an image with random text using the specified font."""

    try:
        # Load the font
        font = ImageFont.truetype(font_path, size=font_size)

        # Create a noise background
        width, height = 280, 36
        noise = numpy.random.randint(195, 255, (height, width), dtype=numpy.uint8)
        noise_image = Image.fromarray(noise, mode='L').filter(ImageFilter.GaussianBlur(radius=1))

        # Create an RGB image and paste the noise background
        image = Image.new("RGB", (width, height), "white")
        image.paste(noise_image, (0, 0))

        # Draw the text
        draw = ImageDraw.Draw(image)

        # Use textbbox to calculate text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height - height_control) // 2
        draw.text((x, y), text, font=font, fill="black")

        # Save the image
        image.save(os.path.join(output_dir, f"{dataset_id}.png"))
        # Save the ground truth text
        with open(os.path.join(output_dir, f"{dataset_id}.gt.txt"), "w") as f:
            f.write(text)
        
    except Exception as e:
        tqdm.write(f"Error with font {font_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", required=True, type=int, help="Quantity of images to generate for each font")
    args = parser.parse_args()

    qty = args.q
    if qty < 1:
        raise ValueError("Quantity must be greater than 0")

    recreate_output_folder(output_dir)
    
    # Generate images for each font and random type
    for font in tqdm(fonts, bar_format=bar_format):
        font_path = font['path']
        font_size = font['size']
        height_control = font['height_control']
        
        for rand_type in tqdm(rand_types, bar_format=bar_format, leave=False):
            for _ in tqdm(range(qty), bar_format=bar_format, leave=False):
                dataset_id, text = rand_type()

                generate_image(dataset_id, text, font_path, font_size, height_control, output_dir)
                
                tqdm.write(f"{font_path} | {dataset_id} | {text} ")
