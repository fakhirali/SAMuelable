from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else "cpu"

model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[1700, 850]]] # 2D localization of a window


inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                     inputs["original_sizes"].cpu(),
                                                     inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores


scores

# plt.imshow(np.array(masks[0][0][-1]))

img = np.array(raw_image)

masks[0].shape

plt.imshow(img)
plt.plot(input_points[0][0][0], input_points[0][0][1], 'ro')
plt.imshow(np.array(masks[0][0][-1]), alpha=0.5)

plt.imshow(img)
plt.plot(1700, 850, 'ro')