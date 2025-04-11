import random
import threading
from ulid import ULID

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

def generate_random_number_string():
    """Generate a random number string in the format NNNNNNNN."""
    
    ulid = ULID()
    nnnnnnnn = str(random.randint(10000000, 99999999)).zfill(8)

    return (str(ulid), f"{nnnnnnnn}")

def generate_random_string(characters = None):
    """Generate a random string of uppercase letters and digits."""

    ulid = ULID()
    if not characters:
        characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    string = ''.join(random.choice(characters) for _ in range(16))

    return (str(ulid), f"{string}")