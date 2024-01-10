from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request, render_template
import os

app = Flask(__name__)

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else "cpu"

model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


def return_mask(img, input_points):
    inputs = processor(img, input_points=input_points, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                         inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())
    mask = np.array(masks[0][0][-1])
    return mask


raw_image = Image.open('static/img.jpg').convert("RGB")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', file='img.jpg')
global_mask = np.zeros((raw_image.size[1], raw_image.size[0]), dtype=np.uint8)
@app.route('/run_model', methods=['GET'])
def run_model():
    global global_mask
    x = request.args.get('x')
    y = request.args.get('y')
    mask = return_mask(raw_image, [[[x, y]]])
    orig_img = np.array(raw_image)
    mask = np.array(mask, dtype=np.uint8) * 1
    global_mask += mask
    mask = global_mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = mask.reshape(mask.shape[0], -1, 1)
    mask_img = np.concatenate((mask * 255, mask * 0, mask * 0), 2)
    mask_white_img = np.concatenate((mask * 255, mask * 255, mask * 255), 2)
    cv2.imwrite('static/mask.png', mask_white_img)
    final_img = cv2.addWeighted(orig_img, 1, mask_img, 0.5, 0)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/result.png', final_img)
    return jsonify({'image_path': '/static/result.png'})


if __name__ == '__main__':
    app.run(debug=False)

#
# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# input_points = [[[1700, 850]]] # 2D localization of a window
#
# mask = return_mask(raw_image, input_points)
#
# img = np.array(raw_image)
# plt.imshow(img)
# plt.plot(input_points[0][0][0], input_points[0][0][1], 'ro')
#
# plt.imshow(mask, alpha=mask*0.5)
#
# plt.imshow(img)
# plt.plot(1700, 850, 'ro')
