"""
Logging the model information, including the parameters and the structure.
"""
import os

from torchinfo import summary


def save_model_structure(model_structure, model_params_name, model_summary, log_config):
    """Save the model structure to the file."""

    model_log_path = log_config.logging_path
    to_save_dir = model_log_path
    os.makedirs(to_save_dir, exist_ok=True)
    model_path = os.path.join(to_save_dir, "detailed_model_structure.log")
    condense_path = os.path.join(to_save_dir, "condense_model_structure.log")
    summary_path = os.path.join(to_save_dir, "model_summary_structure.log")

    if not os.path.exists(model_path):
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(model_structure)

    if not os.path.exists(condense_path):
        with open(condense_path, "w", encoding="utf-8") as f:
            for item in model_params_name:
                f.write(f"{item}\n")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(str(model_summary))


class ModelInformationLogging:
    """The normal model logging."""

    def __init__(self, defined_model, log_config, max_depth: int = 1):
        self.log_results = summary(
            defined_model, depth=max_depth, col_names=["kernel_size", "num_params"]
        )

        # Save the detailed model structure to files
        model_structure = str(defined_model)
        model_params_name = list(defined_model.state_dict().keys())

        save_model_structure(
            model_structure, model_params_name, self.log_results, log_config=log_config
        )
