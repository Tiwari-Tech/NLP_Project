
import os
from pathlib import Path

from NLP_Project.logging import logger
from NLP_Project.entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            dataset_dir = Path(
                "artifacts/data_ingestion/samsum_dataset"
            )

            # Check whether the extracted dataset directory exists.
            if not dataset_dir.is_dir():
                logger.error(
                    f"Dataset directory not found: {dataset_dir}"
                )
                validation_status = False
            else:
                # Check that every required file or directory exists.
                all_files = {
                    item.name for item in dataset_dir.iterdir()
                }

                missing_files = [
                    name
                    for name in self.config.ALL_REQUIRED_FILES
                    if name not in all_files
                ]

                validation_status = len(missing_files) == 0

                if missing_files:
                    logger.error(
                        f"Missing required files/directories: "
                        f"{missing_files}"
                    )
                else:
                    logger.info(
                        "All required dataset files/directories exist."
                    )

            # Ensure the status file's parent directory exists.
            status_file = Path(self.config.STATUS_FILE)
            status_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save validation status.
            with status_file.open("w", encoding="utf-8") as file:
                file.write(
                    f"Validation status: {validation_status}\n"
                )

            logger.info(
                f"Data validation status: {validation_status}"
            )

            return validation_status

        except Exception as e:
            logger.exception(
                f"Error during data validation: {e}"
            )
            raise
