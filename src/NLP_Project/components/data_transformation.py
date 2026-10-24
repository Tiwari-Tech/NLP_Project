
import os
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from NLP_Project.logging import logger
from NLP_Project.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        logger.info(
            f"Loading tokenizer: {self.config.tokenizer_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name
        )

    def convert_examples_to_features(self, example_batch):
        # Tokenize input dialogues
        input_encodings = self.tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True
        )

        # Tokenize target summaries
        target_encodings = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True
        )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]
        }

    def convert(self):
        try:
            data_path = Path(self.config.data_path)
            output_path = (
                Path(self.config.root_dir) / "samsum_dataset"
            )

            # Verify that the input dataset exists
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at: {data_path}"
                )

            logger.info(
                f"Loading dataset from: {data_path}"
            )

            dataset_samsum = load_from_disk(str(data_path))

            logger.info(
                "Starting dataset tokenization..."
            )

            dataset_samsum_pt = dataset_samsum.map(
                self.convert_examples_to_features,
                batched=True,
                desc="Tokenizing SAMSum dataset"
            )

            # Create output directory
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            logger.info(
                f"Saving transformed dataset to: {output_path}"
            )

            dataset_samsum_pt.save_to_disk(str(output_path))

            logger.info(
                "Data transformation completed successfully."
            )

            return dataset_samsum_pt

        except Exception as e:
            logger.exception(
                f"Error during data transformation: {e}"
            )
            raise
