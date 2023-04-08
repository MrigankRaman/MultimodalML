from models import *
from pretrain_data import *
from transformers import AutoTokenizer, AutoProcessor,Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW
from test import *
import ipdb
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

# def CustomTrainer(Seq2SeqTrainer):
#     def __init__(self, train_size, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.args = args
#         self.train_size = train_size
#     def create_scheduler(self, num_training_steps, optimizer):
#         scheduler_steplr = StepLR(optimizer, step_size=5*self.train_size//self.args.per_device_train_batch_size, gamma=0.1)
#         scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=100, after_scheduler=scheduler_steplr)
#         return scheduler


def pretrainer(args):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14").image_processor
    tokenizer.add_tokens("[IMAGE1]")
    tokenizer.add_tokens("[IMAGE2]")
    tokenizer.image1_token_id = tokenizer.convert_tokens_to_ids("[IMAGE1]")
    tokenizer.image2_token_id = tokenizer.convert_tokens_to_ids("[IMAGE2]")
    model = FlanLimber(args, tokenizer)
    model.lm.resize_token_embeddings(len(tokenizer))
    train_dataset = CapDataset('/data/mrigankr/mscoco/', tokenizer, image_processor, max_len = 64, n_visual_tokens = args.n_visual_tokens)
    interval = 1
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        report_to="none",
        logging_steps = 100,
        # logging_strategy="epoch",
        save_steps = interval,
        #https://github.com/huggingface/transformers/blob/504db92e7da010070c36e185332420a1d52c12b2/src/transformers/trainer.py#L626
        label_names = ['labels'],
        seed = args.seed,
        bf16=True, deepspeed= None
    )
    trainer = Seq2SeqTrainer( model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()