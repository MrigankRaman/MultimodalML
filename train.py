from models import *
from datasets import *

from transformers import get_polynomial_decay_schedule_with_warmup
from transformers.optimization import AdamW
from test import *
import ipdb

def trainer(args):
    model = ModelWrapper(args.model)
    if hasattr(model.model.config, 'image_size'):
        image_size = model.model.config.image_size
    else:
        image_size = 224
    train_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/train.json', args.model, image_size = image_size, image_path = "/data/mrigankr/mml/train/")
    eval_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/dev.json', args.model, image_size = image_size, image_path = "/data/mrigankr/mml/dev/")
    interval = args.interval
    interval = interval//torch.cuda.device_count()
    def accuracy_metric(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        # save_strategy="epoch",
        report_to="none",
        evaluation_strategy = "steps",
        eval_steps=200,
        logging_steps = 100,
        # logging_strategy="epoch",
        save_steps = interval,
        metric_for_best_model = "eval_accuracy",
        greater_is_better = True,
        #https://github.com/huggingface/transformers/blob/504db92e7da010070c36e185332420a1d52c12b2/src/transformers/trainer.py#L626
        label_names = ['labels'],
        load_best_model_at_end=True,
        seed = args.seed,
        save_total_limit = 2,
        fp16=False, deepspeed= None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        # tokenizer=tokenizer,
        data_collator=None,
        compute_metrics = accuracy_metric
    )
    trainer.train()
# img2dataset --url_list mscoco.parquet --input_format "parquet"\
#          --url_col "URL" --caption_col "TEXT" --output_format webdataset\
#            --output_folder /data/mrigankr/mscoco --processes_count 16 --thread_count 64 --image_size 256\
#              --enable_wandb False