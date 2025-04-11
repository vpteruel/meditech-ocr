import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-char", required=False, type=str, help="Characters to use for random string generation")
    args = parser.parse_args()

    char = args.char

    for font in constants.FONTS:
        image_id = get_next_image_id()
        _, text = generate_random_string(characters=char)

        font_path = font['path']
        font_size = font['size']
        charset_boxing = font['charset_boxing']
        output_dir = "tesstrain/data/Meditech-ground-truth"
        debug = True

        # Call the generate_image function
        _, _, _ = generate_image(image_id, text, font_path, font_size, charset_boxing, output_dir, debug)