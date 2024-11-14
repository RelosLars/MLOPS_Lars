from datetime import datetime
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import wandb
import random
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from optuna.integration.wandb import WeightsAndBiasesCallback
import yaml
import os
import argparse

print(torch.cuda.is_available())

class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    
class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        optimizer_type: str,
        momentum: float,
        beta1: float,
        beta2: float,
        train_batch_size: int = 64,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        # Create a new WandbLogger for this trial

        # log_optimizer_hparams(model=self)

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()



    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.optimizer_type == 'adamw':
          optimizer = torch.optim.AdamW(
              optimizer_grouped_parameters,
              lr=self.hparams.learning_rate,
              weight_decay=self.hparams.weight_decay
          )
        elif self.hparams.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.hparams.optimizer_type}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]







def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed)


with open("/app/config.yaml", "r") as file:
    config = yaml.safe_load(file)


default_hyperparameters  = config["hyperparameters"]

parser = argparse.ArgumentParser(description="Train BERT model with configurable hyperparameters")

# Hyperparameters from command-line arguments
hyperparams_group = parser.add_argument_group('Hyperparameters')
hyperparams_group.add_argument("--learning_rate", type=float, default=default_hyperparameters.get("learning_rate"),
                    help="Learning rate for the optimizer")
hyperparams_group.add_argument("--warmup_steps", type=int, default=default_hyperparameters.get("warmup_steps"),
                    help="Number of warmup steps")
hyperparams_group.add_argument("--weight_decay", type=float, default=default_hyperparameters.get("weight_decay"),
                    help="Weight decay for the optimizer")
hyperparams_group.add_argument("--optimizer_type", type=str, default=default_hyperparameters.get("optimizer_type"),
                    help="Optimizer type (e.g., 'sgd', 'adam')")
hyperparams_group.add_argument("--momentum", type=float, default=default_hyperparameters.get("momentum"),
                    help="Momentum factor (only used if optimizer_type is 'sgd')")
hyperparams_group.add_argument("--beta1", type=float, default=default_hyperparameters.get("beta1"),
                    help="Beta1 parameter for the Adam optimizer")
hyperparams_group.add_argument("--beta2", type=float, default=default_hyperparameters.get("beta2"),
                    help="Beta2 parameter for the Adam optimizer")

wandb_group = parser.add_argument_group('WandB Configuration')
wandb_group.add_argument("--api_key", type=str, default=config["wandb"]["api_key"],
                    help="api key for wandb")
wandb_group.add_argument("--project", type=str, default=config["wandb"]["project"],
                    help="project name for wandb")
wandb_group.add_argument("--run_name", type=str, default=config["wandb"]["run_name"],
                    help="run name for wandb")


model_group = parser.add_argument_group('Model Configuration')
model_group.add_argument("--epochs", type=int, default=config["model"]["epochs"],
                    help="number of epochs for the training run")
model_group.add_argument("--save_path", type=str, default=config["model"]["save_path"],
                    help="save path for the model")

args = parser.parse_args()


wandb_kwargs = {
    "project": args.project,
    "name": args.run_name 
}
wandb_enabled = args.api_key != ""
if wandb_enabled:
    wandb.login(key=args.api_key)

wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

# Consolidate hyperparameters from command-line or config
hyperparameters = {
    "learning_rate": args.learning_rate,
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    "optimizer_type": args.optimizer_type,
    "momentum": args.momentum,
    "beta1": args.beta1,
    "beta2": args.beta2,
}

print(f"hyperparameters:\n{hyperparameters}")

seed = 42
set_seed(seed)

# Initialize Wandb logger
if wandb_enabled:
    wandb_logger = WandbLogger(
        project=config["wandb"]["project"],
        name=f"single-hp-tuning-seed-{seed}",
        log_model="all"
    )

# Load data
dm = GLUEDataModule(
    model_name_or_path="distilbert-base-uncased",
    task_name="mrpc",
)
dm.setup("fit")

# Initialize model with the fixed hyperparameters
model = GLUETransformer(
    model_name_or_path="distilbert-base-uncased",
    num_labels=dm.num_labels,
    task_name=dm.task_name,
    learning_rate=hyperparameters["learning_rate"],
    weight_decay=hyperparameters["weight_decay"],
    optimizer_type=hyperparameters["optimizer_type"],
    momentum=hyperparameters["momentum"],
    beta1=hyperparameters["beta1"],
    beta2=hyperparameters["beta2"],
    warmup_steps=hyperparameters["warmup_steps"],
)

# Train the model
if torch.cuda.is_available():
    if wandb_enabled:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
        )
    else:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1,
        )
else:
    if wandb_enabled:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            logger=wandb_logger,
            accelerator="cpu",
            devices=1,
        )
    else:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="cpu",
            devices=1,
        )

trainer.fit(model, dm)

# Validate the model and log validation loss to Wandb
val_result = trainer.validate(model, datamodule=dm)
val_loss = val_result[0]['val_loss']
if wandb_enabled:
    wandb.log({"val_loss": val_loss})

# Log hyperparameters and results
print("Run completed with hyperparameters: ", hyperparameters)
print("Validation loss: ", val_loss)

# Finish Wandb run
if wandb_enabled:
    wandb.finish()



model_save_path = args.save_path
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")