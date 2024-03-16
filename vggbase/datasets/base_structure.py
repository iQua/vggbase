"""
The implementation of data storing structure of VGBase.
"""

import os
import logging
from collections import defaultdict

from vggbase.datasets.datalib import data_utils
from vggbase.config import Config


class DataSourceStructure:
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """

    def __init__(self):
        # data name
        self.data_name = ""
        self.data_path = ""
        # source data name
        self.source_data_name = ""
        # the root folder where the source data
        # is stored
        self.source_data_path = ""
        # the directory of the source data
        self.source_data_dir_path = ""
        # the text name of the contained modalities
        self.supported_modalities = []

        # define the information container for the source data
        #  - source_data_path: the original downloaded data
        #  - base_data_path: the source data used for the model
        # For some datasets, we directly utilize the base_data_path as
        #  there is no need to process the original downloaded data to put them
        #  in the base_data_path dir.
        self.data_info = {
            "source_data_path": "",
            "source_data_dir_path": "",
            "base_data_path": "",
        }
        # the data types and corresponding file format
        self.source_data_types = []
        self.source_data_file_formats = []

        # define the paths for the splited root data - train, test, and val
        self.splits = ["train", "val", "test"]
        self.splits_info = defaultdict(dict)

        self.meta_datacatalog_filename = "vgbase_metadata"
        self.datacatalog_filename = "vgbase_datacatalog"

        self.prepare_base_data()
        self.prepare_source_data()

    def prepare_base_data(self):
        """Preprae the base data."""

        # prepare the base data
        self.data_name = Config().data.data_name
        self.data_path = Config().data.data_path
        self.set_base_data_path(data_path=self.data_path, base_data_name=self.data_name)

    def prepare_source_data(self):
        """Prepare the source data."""

        self.source_data_name = Config().data.datasource_name
        self.source_data_path = Config().data.datasource_path
        self.source_data_dir_path = os.path.join(
            self.source_data_path, self.source_data_name
        )
        source_data_download_id = Config().data.datasource_download_address
        if source_data_download_id is not None:
            data_utils.download_google_driver_data(
                download_file_id=source_data_download_id,
                extract_download_file_name=self.source_data_name,
                put_data_dir=self.source_data_path,
            )
        self.connect_source_data(self.source_data_dir_path)

    def set_base_data_path(
        self, data_path, base_data_name
    ):  # the directory name of the working data
        """Generate the data structure based on the defined data path"""
        base_data_path = os.path.join(data_path, base_data_name)

        if not os.path.exists(base_data_path):
            os.makedirs(base_data_path)
        self.data_info["base_data_path"] = base_data_path

    def build_source_data_structure(self):
        """Build the whole data structure."""
        # extract the data information and structure
        self.data_info["source_data_path"] = self.source_data_path
        self.data_info["source_data_dir_path"] = self.source_data_dir_path

        for type_name, type_format in zip(
            self.source_data_types, self.source_data_file_formats
        ):
            type_data_path = os.path.join(self.source_data_dir_path, type_name)
            self.data_info[type_name] = dict()
            self.data_info[type_name]["path"] = type_data_path
            self.data_info[type_name]["format"] = type_format
            self.data_info[type_name]["num_samples"] = len(os.listdir(type_data_path))

    def build_splits_structure(self):
        # generate path/type information for splits
        for split_type in self.splits:
            self.splits_info[split_type]["path"] = self.source_data_dir_path
            self.splits_info[split_type]["split_file"] = os.path.join(
                self.source_data_dir_path, split_type + ".txt"
            )

    def connect_source_data(self, source_data_path):
        """Get the path where data is stored."""

        if data_utils.is_data_exist(source_data_path):
            logging.info(
                "Successfully connected the source data from the path %s.",
                source_data_path,
            )
        else:
            logging.info(
                "Fail to connect the source data from the path %s.",
                source_data_path,
            )

    def set_modality_format(self, modality_name):
        """An interface to set the modality name
        Thus, calling this func to obtain the modality name
         in all parts of the class to achieve the consistency
        """
        if modality_name in ["rgb", "flow"]:
            modality_format = "rawframes"
        else:  # convert to plurality
            modality_format = modality_name + "s"

        return modality_format

    def set_modality_path_key_format(self, modality_name):
        """An interface to set the modality path
        Thus, calling this func to obtain the modality path
         in all parts of the class to achieve the consistency
        """
        modality_format = self.set_modality_format(modality_name)

        return modality_format + "_" + "path"

    def get_metadatacatalog_path(self):
        """Get the path for meta datacatalog."""
        root_path = self.data_info["base_data_path"]
        return os.path.join(root_path, f"{self.meta_datacatalog_filename}.json")

    def get_split_datacatalog_path(self, split_type):
        """Get the split datacatalog."""
        root_path = self.data_info["base_data_path"]
        datacatalog_filename = f"{split_type}_{self.datacatalog_filename}"
        return os.path.join(root_path, f"{datacatalog_filename}.json")
