from typing import List

import base64
# from fastapi import FastAPI
# from pydantic import BaseModel
import os
import io

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import torch

import numpy as np ###Add to dockerfile
import open_clip

class VLMManager:
    def __init__(self):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fetch the OD model
        model_directory = os.getenv("MODEL_PATH", "/usr/src/models")
        self.infer = YOLO("models/yolov8-smol-25epochs.pt") # for testing
        self.infer.to(self.device)

        # Fetch the CLIP model
        clip_model, pdata = ('ViT-H-14-quickgelu', 'dfn5b')
        clip_path = "models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
        path = "models/DFN5B-CLIP-ViT-H-14-378"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_path,
                                                            image_interpolation="bicubic",
                                                            image_resize_mode="squash")
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        
        
                                                       
    # def _identify(self, image: bytes, caption: str) -> List[int]:
    #     # perform object detection with a vision-language model
    #     return [0, 0, 0, 0]
                                        
    def identify(self, image_bytes: bytes, query: str) -> List[int]:

        def decode(image_bytes):
            im = Image.open(io.BytesIO(image_bytes))
            return im

        ### Using OD and CLIP, get the bbox given a PIL image and a caption
        def detect_objects(image):
            res = self.infer.predict(image)
            bboxes = res[0].boxes.xyxy
            
            if bboxes.nelement() == 0:
                bboxes = torch.tensor([[1, 1, 1000, 500]])
                
            return bboxes

        def cropped_images(image, bboxes):
            cropped_images = []
            for bbox in bboxes:
                bbox = [int(x) for x in bbox]
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                cropped_images.append(cropped)
            return cropped_images

#         def clip_query(images, query):
            
#             inputs = self.processor(text=[query], images=images, return_tensors="pt", padding=True).to(self.device)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             return outputs.logits_per_image
        
        def clip_query(images, query):
            images = [self.preprocess(img) for img in images]
            image_input = torch.tensor(np.stack(images)).to(self.device)  # Move image input to GPU
            
            # text_tokens = self.tokenizer([query]).to(self.device)  # Move text tokens to GPU
            
            text_tokens = self.tokenizer.batch_encode_plus(
                [query],
                padding='max_length',
                max_length=77,  # Set to the desired maximum length
                truncation=True,  # Ensure truncation if inputs are longer than max_length
                return_tensors='pt'  # Optional: to return as PyTorch tensors, can be 'tf' for TensorFlow tensors
            )["input_ids"].to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input).float()
                text_features = self.model.encode_text(text_tokens).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T  # Move result back to CPU

            return np.argmax(similarity.flatten())

        def softmax(x):
            return torch.nn.functional.softmax(x, dim=0)


        
        image = decode(image_bytes)
        bboxes = detect_objects(image)
        images = cropped_images(image, bboxes)

        id = clip_query(images, query)

        bbox = bboxes[id].tolist()
        ans = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        final = [int(x) for x in ans]
        
        return final
    
