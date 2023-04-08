from __future__ import absolute_import


import random, params, os
import numpy as np
import torch
#test

def set_seeds(seed, reproduce = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    
    # Load Params from CLI / Config File
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    set_seeds(args.seed)
    root = f"./models/{args.model}"

    model_dir = f"{root}/model_{args.model_id}"
    args.model_dir = model_dir
    if args.mode == "train":
        from train import trainer
        trainer(args)
    elif args.mode == "test":
        from test import eval
        eval(args)
    elif args.mode == "pretrain":
        from pretrain import pretrainer
        pretrainer(args)