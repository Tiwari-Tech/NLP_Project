
from NLP_Project.config.configuration import ConfigurationManager
from NLP_Project.components.data_validation import DataValidation
from NLP_Project.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()

            data_validation_config = (
                config.get_data_validation_config()
            )

            data_validation = DataValidation(
                config=data_validation_config
            )

            validation_status = (
                data_validation.validate_all_files_exist()
            )

            if not validation_status:
                raise ValueError(
                    "Data validation failed. "
                    "Check the dataset and status file."
                )

            logger.info(
                "Data validation completed successfully."
            )

        except Exception as e:
            logger.exception(
                f"Error in Data Validation stage: {e}"
            )
            raise
