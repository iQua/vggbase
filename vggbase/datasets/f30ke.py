"""
The Flickr30K Entities dataset.

The data structure and setting follow:
 "http://bryanplummer.com/Flickr30kEntities/".

We utilize the official splits that contain:
 - train: 29783 images,
 - val: 1000 images,
 - test: 1000 images

The file structure of this dataset is:
 - Images (jpg): the raw images
 - Annotations (xml): the bounding boxes
 - Sentence (txt): captions of the image

The data structure under the 'data/' is:
├── Flickr30KEntities           # root dir of Flickr30K Entities dataset
│   ├── Flickr30KEntitiesRaw    # Raw images/annotations and the official splits
│   ├── train     # data dir for the train phase
│   │   └── train_Annotations
│   │   └── train_Images
│   │   └── train_Sentences
│   └── test
│   └── val
"""

import logging

from vggbase.datasets import multimodal_base
from vggbase.datasets.datalib import flickr30kE_utils


class Flickr30KEDataset(multimodal_base.MultiModalDataset):
    """Prepare the Flickr30K Entities dataset."""

    def __init__(
        self,
        datacatalog_filepath,
        image_transform=None,
        text_transform=None,
    ):
        super().__init__(datacatalog_filepath, image_transform, text_transform)
        self.supported_modalities = ["image", "text"]


class DataSource(multimodal_base.MultiModalDataSource):
    """The Flickr30K Entities dataset."""

    def __init__(self):
        super().__init__()

        self.supported_modalities = ["image", "text"]

        # define the path of different data source,
        #   the annotation is .xml, the sentence is in .txt
        self.source_data_types = ["Images", "Annotations", "Sentences"]
        self.source_data_file_formats = ["jpg", "xml", "txt"]

        self.build_source_data_structure()
        self.build_splits_structure()

        # generate the splits information txt for further utilization
        logging.info("Integrating VGGbase datacatalog...")
        flickr30kE_utils.integrate_data(
            data_info=self.data_info,
            splits_info=self.splits_info,
            data_types=self.source_data_types,
            metadata_filepath=self.get_metadatacatalog_path(),
            datacatalog_filename_fn=self.get_split_datacatalog_path,
        )
        self.compute_splits_data_statistics()

    def get_phase_dataset(self, phase: str, image_transform, text_transform):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
        phase_datacatalog_filepath = self.get_split_datacatalog_path(phase)
        dataset = Flickr30KEDataset(
            datacatalog_filepath=phase_datacatalog_filepath,
            image_transform=image_transform,
            text_transform=text_transform,
        )
        return dataset
