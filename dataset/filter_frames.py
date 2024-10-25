from PIL import Image
import imagehash
import os
import argparse

def remove_duplicates(image_folder, hash_size=8):
    image_hashes = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)
            img_hash = imagehash.average_hash(img, hash_size=hash_size)
            if img_hash in image_hashes:
                os.remove(image_path)
                print(f"Removed duplicate frame: {filename}")
            else:
                image_hashes[img_hash] = filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a folder and generate captions.")
    parser.add_argument("--frames_folder", default="data/frames_output", help="Path to folder containing frame files")
    args = parser.parse_args()

    remove_duplicates(args.frames_folder)
