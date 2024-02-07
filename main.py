from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import transformers
import torch
import evaluate
from pprint import pprint 

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

transformers.set_seed(1337)

accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1") # f1 with micro-averaging. Vide: compute_metrics function
model_checkpoint = "google/vit-base-patch16-224-in21k"
batch_size = 8

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

dataset = load_dataset("imagefolder", data_dir="./dataset/HF_DS",  drop_metadata=True, drop_labels=False)

labels = dataset['train'].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label



normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch



model_name = model_checkpoint.split("/")[-1]



model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)




args = TrainingArguments(
    f"{model_name}-finetuned-mgasior-07-02-2024",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-05,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    push_to_hub=True,
)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


import numpy as np


splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.

def compute_metrics(eval_pred):
    """Computes **MICRO** f1 measure on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return f1_metric.compute(references=eval_pred.label_ids, predictions=predictions, average="micro")

# replace def above if you want to calc acc
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


from PIL import Image
from transformers import pipeline
from os import listdir

pipe = pipeline("image-classification", 
                model=model,
                feature_extractor=image_processor)

# from os.path import isfile, join
# clippers = [join('./dataset/HF_DS/clippers',f) for f in listdir('./dataset/HF_DS/clippers') if isfile(join('./dataset/HF_DS/clippers', f))]

# for f_path in clippers:
#     print(pipe(f_path))

trainer.push_to_hub()

# additionals todo: grid search for best parameters (when I have too much free time)
# def model_init(trial):
#     return AutoModelForImageClassification.from_pretrained(
#         model_checkpoint,
#             label2id=label2id,
#             id2label=id2label,
#     )


# def optuna_hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
#         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
#     }

# best_trials = trainer.hyperparameter_search(
#     direction=["minimize", "maximize"],
#     backend="optuna",
#     hp_space=optuna_hp_space,
#     n_trials=20,
# )
