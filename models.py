from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoImageProcessor, ViTModel, ViTMAEModel, ViltModel, AutoProcessor, FlavaModel, CLIPVisionModel
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import json
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import numpy as np
from torchvision.models import resnet50
import ipdb
class ModelWrapper(nn.Module):
    def __init__(self, model_type, args= None, tokenizer = None):
        super(ModelWrapper, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "vilt":
            # config = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2").config
            # self.model = ViltForImagesAndTextClassification(config)
            # ipdb.set_trace()
            self.model = ViltForNLVR2().to(self.device)
            for param in self.model.vilt.parameters():
                param.requires_grad = False
            # self.model.vilt.config = config
        elif model_type == "flava":
            self.model = FlavaForNLVR2().to(self.device)
        elif model_type == "roberta":
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large")
        elif model_type == "vit":
            self.model = ViTforNLVR2()
        elif model_type == "resnet50":
            self.model = ResnetforNLVR2().to(self.device)
        elif model_type == "mae":
            self.model = MAEforNLVR2()
        elif model_type == "roberta_vit":
            self.model = ViTAndRobertaForNLVR2().to(self.device) 
            for param in self.model.vit.parameters():
                param.requires_grad = False
            for param in self.model.roberta.parameters():
                param.requires_grad = False
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        elif model_type == "roberta_mae":
            self.model = MAEAndRobertaForNLVR2().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        elif model_type == "vilt_pretrained":
            self.model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        else:
            print("model not recognized")
        self.model_type = model_type
    def forward(self, input_ids=None, attention_mask=None, pixel_values_1=None, pixel_values_2=None, pixel_values=None, labels = None):
        if self.model_type == "vilt" :
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values_1 = pixel_values_1.to(self.device)
            pixel_values_2 = pixel_values_2.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask = attention_mask, pixel_values_1=pixel_values_1, pixel_values_2=pixel_values_2)
        elif self.model_type == "vilt_pretrained":
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask = attention_mask, pixel_values=pixel_values).logits
        elif self.model_type == "roberta":
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            labels = labels.to(self.model.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits 
        elif self.model_type == "vit" or self.model_type == "resnet50" or self.model_type == "mae":
            pixel_values_1 = pixel_values_1.to(self.device)
            pixel_values_2 = pixel_values_2.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(pixel_values_1=pixel_values_1, pixel_values_2=pixel_values_2)
        elif self.model_type == "roberta_vit" or self.model_type == "roberta_mae":
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values_1 = pixel_values_1.to(self.device)
            pixel_values_2 = pixel_values_2.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask = attention_mask, pixel_values_1=pixel_values_1, pixel_values_2=pixel_values_2)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        else:
            return logits



class ViTforNLVR2(nn.Module):
    def __init__(self):
        super(ViTforNLVR2, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(self.vit.config.hidden_size*2, 2)
        self.config = self.vit.config
        self.device = self.vit.device
    def forward(self, pixel_values_1, pixel_values_2):
        x1 = self.vit(pixel_values = pixel_values_1).pooler_output
        x2 = self.vit(pixel_values = pixel_values_2).pooler_output
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x




class MAEforNLVR2(nn.Module):
    def __init__(self):
        super(MAEforNLVR2, self).__init__()
        self.mae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.classifier = nn.Linear(self.mae.config.hidden_size*2, 2)
        self.config = self.mae.config
        self.device = self.mae.device
    def forward(self, pixel_values_1, pixel_values_2):
        x1 = self.mae(pixel_values = pixel_values_1).last_hidden_state[:,0] 
        x2 = self.mae(pixel_values = pixel_values_2).last_hidden_state[:,0]
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
class ResnetforNLVR2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512*4*2, 2)
        self.config=None
    def forward(self, pixel_values_1, pixel_values_2):
        x1 = self.encoder(pixel_values_1)
        x2 = self.encoder(pixel_values_2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

class ViTAndRobertaForNLVR2(nn.Module):
    def __init__(self):
        super(ViTAndRobertaForNLVR2, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.roberta = AutoModelForSequenceClassification.from_pretrained("roberta-large").to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size*2 + self.roberta.config.hidden_size, self.vit.config.hidden_size*2 + self.roberta.config.hidden_size),
            nn.LayerNorm(self.vit.config.hidden_size*2 + self.roberta.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.vit.config.hidden_size*2 + self.roberta.config.hidden_size, 2),
        )
        self.config = self.vit.config
        self.device = self.vit.device
    def forward(self, pixel_values_1, pixel_values_2, input_ids, attention_mask):
        x1 = self.vit(pixel_values = pixel_values_1).pooler_output
        x2 = self.vit(pixel_values = pixel_values_2).pooler_output
        x3 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True).hidden_states[-1][:,0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x
class MAEAndRobertaForNLVR2(nn.Module):
    def __init__(self):
        super(MAEAndRobertaForNLVR2, self).__init__()
        self.mae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.roberta = AutoModelForSequenceClassification.from_pretrained("roberta-large").to(self.device)
        # self.classifier = nn.Linear(self.mae.config.hidden_size*2 + self.roberta.config.hidden_size, 2)
        self.classifier = nn.Sequential(
            nn.Linear(self.mae.config.hidden_size*2 + self.roberta.config.hidden_size, self.mae.config.hidden_size*2 + self.roberta.config.hidden_size),
            nn.LayerNorm(self.mae.config.hidden_size*2 + self.roberta.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.mae.config.hidden_size*2 + self.roberta.config.hidden_size, 2),
        )
        self.config = self.mae.config
        self.device = self.mae.device
    def forward(self, pixel_values_1, pixel_values_2, input_ids, attention_mask):
        # ipdb.set_trace()
        x1 = self.mae(pixel_values = pixel_values_1).last_hidden_state[:,0] 
        x2 = self.mae(pixel_values = pixel_values_2).last_hidden_state[:,0]
        x3 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True).hidden_states[-1][:,0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

class ViltForNLVR2(nn.Module):
    def __init__(self):
        super(ViltForNLVR2, self).__init__()
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
        self.vilt.config.num_images = 1
        self.vilt.config.num_labels = 2
        self.classifier = nn.Sequential(
            nn.Linear(self.vilt.config.hidden_size * 2, self.vilt.config.hidden_size * 2),
            nn.LayerNorm(self.vilt.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.vilt.config.hidden_size * 2, self.vilt.config.num_labels),
        )
        self.config = self.vilt.config
        self.device = self.vilt.device
    def forward(self, pixel_values_1, pixel_values_2, input_ids, attention_mask):
        x1 = self.vilt(pixel_values = pixel_values_1, input_ids = input_ids, attention_mask = attention_mask).pooler_output
        x2 = self.vilt(pixel_values = pixel_values_2, input_ids = input_ids, attention_mask = attention_mask).pooler_output
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

class FlavaForNLVR2(nn.Module):
    def __init__(self):
        super(FlavaForNLVR2, self).__init__()
        self.flava = FlavaModel.from_pretrained("facebook/flava-full")
        self.flava.config.num_images = 1
        self.flava.config.num_labels = 2
        self.classifier = nn.Sequential(
            nn.Linear(self.flava.config.hidden_size * 2, self.flava.config.hidden_size * 2),
            nn.LayerNorm(self.flava.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.flava.config.hidden_size * 2, self.flava.config.num_labels),
        )
        self.config = self.flava.config
        self.device = self.flava.device
    def forward(self, pixel_values_1, pixel_values_2, input_ids, attention_mask):
        x1 = self.flava(pixel_values = pixel_values_1, input_ids = input_ids, attention_mask = attention_mask).multimodal_output.pooler_output
        x2 = self.flava(pixel_values = pixel_values_2, input_ids = input_ids, attention_mask = attention_mask).multimodal_output.pooler_output
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

class FlanLimber(nn.Module):
    def __init__(self, args, tokenizer):
        super(FlanLimber, self).__init__()
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.lm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
        self.args = args
        self.input_embeddings = self.lm.get_input_embeddings()
        embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
        self.projection = nn.Linear(self.image_encoder.config.hidden_size, embedding_dim)
        self.image_dropout = nn.Dropout(self.args.image_embed_dropout_prob)
        self.tokenizer = tokenizer
        if args.freeze_lm:
            self.lm.eval()
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()
        if args.freeze_vm:
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            self.image_encoder.train()

    def get_visual_embs(self, pixel_values):
        outputs = self.image_encoder(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.projection(encoder_outputs)
        visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
        visual_embs = self.image_dropout(visual_embs)
        return visual_embs

    def train(self):
        super(FlanLimber, self).train()
        if self.args.freeze_lm:
            self.lm.eval()
        if self.args.freeze_vm:
            self.image_encoder.eval()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        prompt_embs = self.input_embeddings(input_ids)
        if pixel_values is not None:
            if len(list(pixel_values.size())) == 5:
                pixel_values_1 = pixel_values[:,0]
                pixel_values_2 = pixel_values[:,1]
                vis_embs1 = get_visual_embs(pixel_values_1)
                vis_embs2 = get_visual_embs(pixel_values_2)
                prompt_embs[input_ids == self.tokenizer.image1_token_id] = vis_embs1
                prompt_embs[input_ids == self.tokenizer.image2_token_id] = vis_embs2
            else:
                vis_embs = self.get_visual_embs(pixel_values)
                prompt_embs = prompt_embs.type(vis_embs.dtype)
                prompt_embs[input_ids == self.tokenizer.image1_token_id] = vis_embs.reshape(self.args.n_visual_tokens*self.args.n_visual_tokens, -1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.lm(inputs_embeds=prompt_embs,labels=labels, output_hidden_states=True)
        return [outputs.loss]

    def generate(pixel_values = None, input_ids = None, attention_mask = None, **generate_kwargs):
        prompt_embs = self.input_embeddings[input_ids]
        if pixel_values is not None:
            if len(list(pixel_values.size())) == 5:
                pixel_values_1 = pixel_values[:,0]
                pixel_values_2 = pixel_values[:,1]
                vis_embs1 = get_visual_embs(pixel_values_1)
                vis_embs2 = get_visual_embs(pixel_values_2)
                prompt_embs[input_ids == self.tokenizer.image1_token_id] = vis_embs1
                prompt_embs[input_ids == self.tokenizer.image2_token_id] = vis_embs2
            else:
                vis_embs = get_visual_embs(pixel_values)
                prompt_embs[input_ids == self.tokenizer.image1_token_id] = vis_embs
        outputs = self.lm.generate(inputs_embeds=prompt_embs, **generate_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
