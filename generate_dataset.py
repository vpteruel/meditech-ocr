from tqdm import tqdm
import argparse
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import constants
from random_seeds import (
    generate_random_string,
    generate_random_ulid,
    generate_random_date_string,
    generate_random_number_string,
    get_next_image_id,
)
from image_generator import generate_image

# bar_format = "[{l_bar}{bar} {rate_fmt}{postfix} | {n_fmt}/{total_fmt} {elapsed}<{remaining}]"
bar_format = "{l_bar}{bar}|"

# Output directory for generated images
output_dir = "tesstrain/data/Meditech-ground-truth"

def recreate_output_folder(output_dir):
    """Delete and recreate the output directory."""

    if os.path.exists(output_dir):
        print(f"Deleting {output_dir} folder...")
        shutil.rmtree(output_dir)
    print(f"Creating {output_dir} folder...")
    os.makedirs(output_dir, exist_ok=True)

def generate_images_for_font(font, qty, output_dir, progress_bar, debug):
    """Generate images for a specific font."""
    font_path = font['path']
    font_size = font['size']
    charset_boxing = font['charset_boxing']

    for rand_type in rand_types:
        for _ in range(qty):
            _, text = rand_type()
            image_id = get_next_image_id()  # Get sequential ID instead
            _, _, _ = generate_image(image_id, text, font_path, font_size, charset_boxing, output_dir, debug)
            progress_bar.update(1)  # Update the shared progress bar

# List of random types to generate
rand_types = [
    generate_random_ulid,
    generate_random_date_string,
    generate_random_number_string,
    generate_random_string,
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
        total_tasks = len(constants.FONTS) * len(rand_types) * qty

        # Create a shared tqdm progress bar
        with tqdm(total=total_tasks, desc="Generating Images", bar_format=bar_format) as progress_bar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(generate_images_for_font, font, qty, output_dir, progress_bar, debug)
                    for font in constants.FONTS
                ]

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    future.result()
