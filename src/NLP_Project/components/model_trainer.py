
import os
import logging

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from NLP_Project.entity import ModelTrainerConfig


logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # 1. Select the available device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", device)

        # 2. Load the tokenizer
        logger.info("Loading tokenizer: %s", self.config.model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt
        )

        # 3. Load the pretrained model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        )
        model.to(device)

        # 4. Load the transformed dataset
        logger.info(
            "Loading transformed dataset from: %s",
            self.config.data_path,
        )
        dataset = load_from_disk(str(self.config.data_path))

        required_splits = {"train", "validation"}
        available_splits = set(dataset.keys())

        if not required_splits.issubset(available_splits):
            raise ValueError(
                f"Dataset must contain {required_splits}; "
                f"found {available_splits}"
            )

        if len(dataset["train"]) == 0:
            raise ValueError("The training dataset is empty.")

        if len(dataset["validation"]) == 0:
            raise ValueError("The validation dataset is empty.")

        # 5. Prepare the data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
        )

        # 6. Convert configuration values to their expected types
        num_train_epochs = float(self.config.num_train_epochs)
        warmup_steps = int(self.config.warmup_steps)
        train_batch_size = int(
            self.config.per_device_train_batch_size
        )
        weight_decay = float(self.config.weight_decay)
        logging_steps = int(self.config.logging_steps)
        eval_steps = int(self.config.eval_steps)
        gradient_accumulation_steps = int(
            self.config.gradient_accumulation_steps
        )

        if num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be greater than 0.")

        if train_batch_size <= 0:
            raise ValueError(
                "per_device_train_batch_size must be greater than 0."
            )

        if logging_steps <= 0:
            raise ValueError("logging_steps must be greater than 0.")

        if eval_steps <= 0:
            raise ValueError("eval_steps must be greater than 0.")

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than 0."
            )

        # Normalize the evaluation strategy
        evaluation_strategy = str(
            self.config.evaluation_strategy
        ).lower().strip()

        if evaluation_strategy not in {"steps", "epoch", "no"}:
            raise ValueError(
                "evaluation_strategy must be 'steps', 'epoch', or 'no'; "
                f"received {evaluation_strategy!r}"
            )

        # 7. Configure training
        output_dir = str(self.config.root_dir)
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            eval_strategy=evaluation_strategy,
            eval_steps=(
                eval_steps
                if evaluation_strategy == "steps"
                else None
            ),
            save_strategy="no",
            gradient_accumulation_steps=gradient_accumulation_steps,
            report_to="none",
            fp16=torch.cuda.is_available(),
            use_cpu=not torch.cuda.is_available(),
        )

        # 8. Initialize Trainer once
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        # 9. Fine-tune the model
        logger.info("Starting model training...")
        train_result = trainer.train()
        logger.info("Training metrics: %s", train_result.metrics)

        # 10. Save the trained model and tokenizer
        model_output_dir = os.path.join(
            output_dir, "pegasus-samsum-model"
        )
        tokenizer_output_dir = os.path.join(
            output_dir, "tokenizer"
        )

        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(tokenizer_output_dir)

        # 11. Save training metrics
        trainer.save_metrics("train", train_result.metrics)

        logger.info("Model saved to: %s", model_output_dir)
        logger.info("Tokenizer saved to: %s", tokenizer_output_dir)
        logger.info("Model training completed successfully.")
