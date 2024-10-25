from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import pandas as pd
import argparse

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def generate_captions_for_frames(frames_folder):
    # Loop through frames to generate captions
    captions = []
    for filename in os.listdir(frames_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(frames_folder, filename)
            caption = generate_caption(image_path)
            captions.append({"image": filename, "caption": caption})
    return captions

def save_captions(captions, output_csv):
    df = pd.DataFrame(captions)
    df.to_csv(output_csv, index=False)
    print("Captions saved to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a folder and generate captions.")
    parser.add_argument("--frames_folder", default="data/frames_output", help="Path to folder containing frame files")
    parser.add_argument("--output_csv", default="data/image_captions.csv", help="Path to save captions CSV")
    args = parser.parse_args()

    captions = generate_captions_for_frames(args.frames_folder)
    save_captions(captions, args.output_csv)

    print("Caption generation completed.")
