import argparse
from distutils import util
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='NLVR2 Training')
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--mode", help="train/test", type=str, default = "train", choices = ["train","test", "pretrain"])

    parser.add_argument("--model", help="Model Architecture", type=str, default = "roberta", choices = ["roberta", "vit", "vilt", "resnet50", "mae", "roberta_vit", "roberta_mae", "visual_bert"])
    parser.add_argument("--model_id", help = "For Saving", type = str, default = '0')
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--interval", help ="Logging Interval", type = int, default = 1000)
    parser.add_argument("--n_visual_tokens", help ="number of image tokens", type = int, default = 32)
    parser.add_argument("--image_embed_dropout_prob", help ="dropout for image tokens", type = float, default = 0.0)
    parser.add_argument("--freeze_lm", help ="freeze lm", type = bool, default = True)
    parser.add_argument("--freeze_vm", help ="freeze vision", type = bool, default = True)

    #HPARAMS
    parser.add_argument("--num_epochs", help = "Number of Epochs", type = int, default = 8)
    parser.add_argument("--patience", help = "Number of Epochs", type = int, default = 5)
    parser.add_argument("--train_batch_size", help = "Batch Size for Train Set (Default = 32)", type = int, default = 32)
    parser.add_argument("--lr", help = "Learning Rate", type = float, default = 1e-4)
    parser.add_argument("--weight_decay", help = "Weight Decay", type = float, default = 0.01)
    parser.add_argument("--max_length", help = "Max Sequence Length", type = int, default = 512)

    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = "None")
    parser.add_argument("--test_batch_size", help = "Batch Size for Test Set (Default = 32)", type = int, default = 32)
    parser.add_argument("--split", help = "split to test on", type = str, default = "validation", choices = ["train", "validation", "test"])

    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args