"""
An interface used to build the dataset for the visual grounding task
"""

from torch.utils.data import DataLoader

from vggbase.datasets.rig import DataSource as rig_datasource
from vggbase.datasets.f30ke import DataSource as f30ke_datasource

from vggbase.datasets import visual_tranforms
from vggbase.datasets import collate_functions

from vggbase.datasets.language.dynamic_tokenizer import LanguageDynamicTokenizer

datasource_factory = {
    "ReferItGame": rig_datasource,
    "Flickr30K_Entities": f30ke_datasource,
}

collate_fn_factory = {"default": collate_functions.FormatSamplesCreator}


class VGDatasetBuilder:
    """
    A base dataset builder for the visual grounding task.
    """

    def __init__(self, tokenizer: LanguageDynamicTokenizer, data_config: dict):
        super().__init__()

        # The data config
        self.data_config = data_config

        # The tokenizer used to process the text data
        self.language_tokenizer = tokenizer

        # The datasource containing train, val, test splits
        self.datasource = None

        # The collate_fn for the dataloader
        self.collate_fn = None

    def prepare_datasource(self, data_name: str = None):
        """Prepare the data source by defining the desire data."""
        data_name = self.data_config["data_name"] if data_name is None else data_name

        if data_name in datasource_factory:
            self.datasource = datasource_factory[data_name]()
        else:
            raise ValueError(f"No such data source: {data_name}")

    def set_visual_transformation(self, phase):
        """Set transformation here for val/train/test split."""

        # Define data augmentation and transformation
        visual_conversion = visual_tranforms.VisualShapeConversion(
            phase=phase, conversion_config=self.data_config["Conversion"]
        )
        visual_augmentation = visual_tranforms.ViauslContentAugmentations(
            phase=phase, augmentation_config=self.data_config["Augmentation"]
        )

        return visual_tranforms.VisualTransformations(
            content_augmentations=visual_augmentation,
            shape_conversion=visual_conversion,
            box_config=self.data_config["Box"],
        )

    def set_collate_fn(self):
        """Set the collate_fn for the dataloader"""
        collate_fn_name = (
            self.data_config["collate_fn_name"]
            if "collate_fn_name" in self.data_config
            else "default"
        )
        if collate_fn_name in collate_fn_factory:
            self.collate_fn = collate_fn_factory[collate_fn_name](
                self.language_tokenizer
            ).collate_function
        else:
            raise ValueError(f"No such collate function: {collate_fn_name}")

    def train_dataloader(self, batch_size: int = 10):
        """Get the train dataloder"""
        visual_transform = self.set_visual_transformation(phase="train")
        phase_dataset = self.datasource.get_phase_dataset(
            phase="train",
            image_transform=visual_transform,
            text_transform=None,
        )
        data_loader = DataLoader(
            phase_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_config["num_workers"],
        )
        return data_loader

    def val_dataloader(self, batch_size: int = 10):
        """Get the val dataloder"""
        visual_transform = self.set_visual_transformation(phase="val")
        phase_dataset = self.datasource.get_phase_dataset(
            phase="val",
            image_transform=visual_transform,
            text_transform=None,
        )
        data_loader = DataLoader(
            phase_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_config["num_workers"],
        )
        return data_loader

    def test_dataloader(self, batch_size: int = 10):
        """Get the test dataloder"""
        visual_transform = self.set_visual_transformation(phase="test")
        phase_dataset = self.datasource.get_phase_dataset(
            phase="test",
            image_transform=visual_transform,
            text_transform=None,
        )

        data_loader = DataLoader(
            phase_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_config["num_workers"],
        )
        return data_loader
