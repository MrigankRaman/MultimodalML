from typing import Optional, Tuple

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
import glob
import json
from tqdm import tqdm
import random

class CapDataset(Dataset):
    def __init__(self, image_folder, tokenizer, image_processor, train: bool = True, max_len: int = 32, precision: str = 'fp32', image_size: int = 224, n_visual_tokens=32):
        self.image_size = image_size
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.train = train
        self.max_len = max_len
        self.precision = precision
        self.n_visual_tokens = n_visual_tokens
        self.examples = []
        for filename in tqdm(glob.glob(os.path.join(self.image_folder, '*.json'))):
            with open(filename) as infile:
                self.examples.append(json.load(infile))

    def get_image(self, identifier):
        image = Image.open(self.image_folder+"/"+identifier+".jpg")
        image = image.convert("RGB")
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size))
        return image

    def get_instruction(self):
        instructions = []
        instructions.append("You are given an image as input. Describe the image in a sentence. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        instructions.append("Given an image as input caption it. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        instructions.append("Use natural language to describe the input image. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        instructions.append("Describe the input image in a sentence. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        instructions.append("Caption the input image. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        instructions.append("Give your opinion about the input image. Input: The image is " + "[IMAGE1] "*self.n_visual_tokens)
        #choose one instruction at random
        instruction = random.choice(instructions)
        return instruction

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        dict_values = {}
        image = self.get_image(self.examples[index]['key'])
        label = "Output: "+ self.examples[index]['caption']
        prompt = self.get_instruction()
        pixel_values = self.image_processor(image, return_tensors = "pt").pixel_values
        tokenized_labels = self.tokenizer(label, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_len)
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        dict_values["input_ids"] = tokenized_prompt["input_ids"].squeeze(0)
        dict_values["attention_mask"] = tokenized_prompt["attention_mask"].squeeze(0)
        dict_values["labels"] = tokenized_labels["input_ids"].squeeze(0)
        dict_values["pixel_values"] = pixel_values.squeeze(0)
        return dict_values
    