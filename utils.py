"""
Utility functions
"""

import os


def validate_image_path(path):
    """Validate image file path"""
    if not os.path.exists(path):
        return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    return any(path.lower().endswith(ext) for ext in valid_extensions)


def save_text_to_file(text, filename, encoding='utf-8'):
    """Save text to file"""
    with open(filename, 'w', encoding=encoding) as f:
        f.write(text)
    print(f"✅ Saved to {filename}")


def load_image_list_from_file(filename):
    """Load list of image paths from text file"""
    with open(filename, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def print_header(text, char='='):
    """Print formatted header"""
    print(f"\n{char * 60}")
    print(text)
    print(f"{char * 60}\n")


def print_subheader(text, char='-'):
    """Print formatted subheader"""
    print(f"\n{char * 60}")
    print(text)
    print(f"{char * 60}")
