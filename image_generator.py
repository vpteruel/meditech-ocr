from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import os
import numpy as np

def generate_image(image_id, text, font_path, font_size, charset_boxing, output_dir, debug):
    """Generate an image with random text using the specified font."""

    try:
        # Load the font
        font = ImageFont.truetype(font_path, size=font_size, encoding="unic")

        # Set the image size
        width, height = 320, 100
        
        # Create a random noise image
        noise = np.random.randint(205, 255, (height, width), dtype=np.uint8)
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
            # Get character boxing values
            boxing = charset_boxing.get(char, {})
            left_boxing = boxing.get('left', 0)
            top_boxing = boxing.get('top', 0)
            right_boxing = boxing.get('right', 0)
            bottom_boxing = boxing.get('bottom', 0)
            
            # Get character width and bounding box
            char_width = font.getlength(char)
            left = x_offset + left_boxing # Left boxing
            right = x_offset + char_width + right_boxing # Right boxing

            # Tesseract uses bottom-left origin, so invert y-coordinates
            y_top = y - ascent - top_boxing  # Top of the character (PIL y decreases upward)
            y_bottom = y + descent - bottom_boxing  # Bottom of the character

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
        image.save(os.path.join(output_dir, f"eng_{image_id:06d}.tif"))
        # image.save(os.path.join(output_dir, f"eng_{image_id:06d}.png"))

        # Save .box file (same name as image)
        with open(os.path.join(output_dir, f"eng_{image_id:06d}.box"), "w") as f:
            f.write("\n".join(box_entries))

        # Save ground truth text
        with open(os.path.join(output_dir, f"eng_{image_id:06d}.gt.txt"), "w") as f:
            f.write(text)

        return (image, box_entries, text)
        
    except Exception as e:
        tqdm.write(f"Error with font {font_path}: {e}")