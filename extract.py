from transformers import AutoImageProcessor, SwinModel
import torch
import os
import pickle
from PIL import Image

CWD = os.path.dirname(__file__)

def get_encodings(images_dir, image_size):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    imgs_list = os.listdir(images_dir)
    encodings = {}
    for image_path in imgs_list:
        with Image.open(os.path.join(images_dir, image_path)) as img:
            img = img.resize(image_size)
            img = image_processor(img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**img)
            encoding = outputs.last_hidden_state
            encodings[image_path] = encoding
    return encodings

if __name__ == '__main__':
    encodings = get_encodings(os.path.join(CWD, 'data/Images'), (224, 224))
    with open(os.path.join(CWD, "encodings_swin.pkl"), "wb") as f:
        pickle.dump(encodings, f)

