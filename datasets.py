from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoImageProcessor, ViTModel
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import json
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import numpy as np
from torchvision import transforms
from randaug import RandAugment
import os

class NLVR2Dataset(Dataset):
    def _init_(self, path, model_type, image_size=None, image_path = None, image_ft_path = "/data/mrigankr/mml/dev_fts"):
        self.examples = []
        self.image_path = image_path    
        self.image_size = image_size
        if model_type == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        if "roberta" in model_type:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        if "vit" in model_type:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            
        elif "mae" in model_type:
            self.processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        elif model_type == "resnet50":
            self.processor = transforms.Compose([transforms.ToTensor()])
        elif model_type == "visual_bert":
            # import ipdb; ipdb.set_trace()
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Model type not supported")
        self.model_type = model_type
        with open(path) as infile:
            for line in infile:
                example = json.loads(line)
                self.examples.append(example)

        self.image_ft_path = image_ft_path


    def get_image(self, name):
        image = Image.open(self.image_path+name)
        image = image.convert("RGB")
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size))
        image = RandAugment(2,9)(image)
        return image

    def _len_(self):
        return len(self.examples)


    def _getitem_(self, index):
        if self.model_type != "visual_bert":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")
        text = self.examples[index]['sentence']
        label = self.examples[index]['label']
        if label == "True":
            label = 1
        else:
            label = 0
        dict_values = {}
        dict_values["labels"] = torch.tensor(label)


        if self.model_type == "vilt":            
            encoding1 = self.processor(image_1, text, return_tensors="pt", padding="max_length", max_length=40, truncation=True, )
            encoding2= self.processor(image_2, text, return_tensors="pt", padding="max_length", max_length=40, truncation=True, )
            input_ids = encoding1.input_ids.squeeze(0)
            pixel_values1 = encoding1.pixel_values.squeeze(0)
            pixel_values2 = encoding2.pixel_values.squeeze(0)
            dict_values["input_ids"] = input_ids
            dict_values["pixel_values_1"] = pixel_values1
            dict_values["pixel_values_2"] = pixel_values2
            dict_values["attention_mask"] = encoding1.attention_mask.squeeze(0)


        elif self.model_type == "roberta":
            text = self.examples[index]['sentence']
            inputs_dict = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            dict_values["input_ids"] = input_ids.squeeze(0)
            dict_values["attention_mask"] = attention_mask.squeeze(0)

        elif self.model_type == "vit" or self.model_type == "mae":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")
            pixel_values_1 = self.processor(image_1, return_tensors="pt")
            pixel_values_2 = self.processor(image_2, return_tensors="pt")
            dict_values["pixel_values_1"] = pixel_values_1["pixel_values"].squeeze(0)
            dict_values["pixel_values_2"] = pixel_values_2["pixel_values"].squeeze(0)
        elif self.model_type == "resnet50":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")
            image_1 = self.processor(image_1)
            image_2 = self.processor(image_2)
            dict_values["pixel_values_1"] = image_1
            dict_values["pixel_values_2"] = image_2

        elif self.model_type == "roberta_vit" or self.model_type == "roberta_mae":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")

            pixel_values_1 = self.processor(image_1, return_tensors="pt")
            pixel_values_2 = self.processor(image_2, return_tensors="pt")
            text = self.examples[index]['sentence']
            inputs_dict = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            dict_values["input_ids"] = input_ids.squeeze(0)
            dict_values["attention_mask"] = attention_mask.squeeze(0)
            dict_values["pixel_values_1"] = pixel_values_1["pixel_values"].squeeze(0)
            dict_values["pixel_values_2"] = pixel_values_2["pixel_values"].squeeze(0)
        elif self.model_type == "visual_bert":
            image_fts_1 = os.path.join(self.image_ft_path, self.examples[index]['identifier'][:-2]+"-img0.npy")
            image_fts_2 = os.path.join(self.image_ft_path, self.examples[index]['identifier'][:-2]+"-img1.npy")
            image_fts_1 = np.load(image_fts_1)
            image_fts_2 = np.load(image_fts_2)
            image_fts_1 = image_fts_1[:144]
            image_fts_2 = image_fts_2[:144]
            image_fts_1 = torch.tensor(image_fts_1)
            image_fts_2 = torch.tensor(image_fts_2)
            #concat the two images
            image_fts = torch.cat((image_fts_1, image_fts_2), dim=0)
            visual_embeds = image_fts
            text = self.examples[index]['sentence']
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            #squeeze inputs
            for key in inputs:
                inputs[key] = inputs[key].squeeze(0)
            visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
            #set the second image token type ids to 2
            # visual_token_type_ids[:image_fts_1.shape[0]] = 2
            # visual_token_type_ids[144:] = 1
            #convert to torch
            # visual_token_type_ids = torch.tensor(visual_token_type_ids)

            # visual_token_type_ids[144:-1] = 2
            # print(visual_token_type_ids)
            visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
            inputs.update({"visual_embeds": visual_embeds,"visual_token_type_ids": visual_token_type_ids,"visual_attention_mask": visual_attention_mask})
            dict_values["inputs"] = inputs
        else:
            print("Model type not supported")
        return dict_values