"""

Although the name of this dataset is referitgame, it actually contains four datasets:
 - ReferItGame http://tamaraberg.com/referitgame/ .
 Then, refer-based datasets http://vision2.cs.unc.edu/refer/:
 - RefCOCO
 - RefCOCO+
 - RefCOCOg

The current dataset is based on the source images of COCO2017/train.

The corresponding names of these datasets in this api are:
    ReferItGame : refclef
    RefCOCO     : refcoco
    RefCOCO+     : refcoco+
    RefCOCOg     : refcocog


The 'split_config' needed to be set to support the following datasets:
- referitgame: 130,525 expressions for referring to 96,654 objects in 19,894 images.
                The samples are splited into three subsets.  train/54,127 referring expressions.
                test/5,842, val/60,103 referring expressions.
- refcoco: 142,209 refer expressions for 50,000 objects.
- refcoco+: 141,564 expressions for 49,856 objects.
- refcocog (google):  25,799 images with 49,856 referred objects and expressions.

The output sample structure of this data is consistent with that
 in the flickr30k entities dataset.


# locate your own data_root, and choose the dataset_splitBy you want to use
refer = REFER(data_root, dataset='refclef',  splitBy='unc')
refer = REFER(data_root, dataset='refclef',  splitBy='berkeley') # 2 train and 1 test images missed
refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
refer = REFER(data_root, dataset='refcoco',  splitBy='google')
refer = REFER(data_root, dataset='refcoco+', splitBy='unc')
refer = REFER(data_root, dataset='refcocog', splitBy='google')   # test split not released yet
refer = REFER(data_root, dataset='refcocog', splitBy='umd')      # Recommended, including train/val/test



"""

import os
import logging
import json


from vggbase.config import Config
from vggbase.datasets.multimodal_base import MultiModalDataset, MultiModalDataSource

from vggbase.datasets.datalib import refer_api
from vggbase.datasets.datalib import refer_utils
from vggbase.datasets.datalib import data_utils


class ReferItGameDataset(MultiModalDataset):
    """Prepares the ReferItGame dataset."""

    def __init__(
        self,
        datacatalog_filepath: str,
        image_transform=None,
        text_transform=None,
    ):
        super().__init__(datacatalog_filepath, image_transform, text_transform)
        self.supported_modalities = ["image", "text"]


class DataSource(MultiModalDataSource):
    """The ReferItGame dataset."""

    def __init__(self):
        super().__init__()

        self.supported_modalities = ["image", "text"]

        self.sub_datasets_name = ["refclef", "refcoco", "refcoco+", "refcocog"]
        self.splits = ["train", "val", "test", "testA", "testB", "testC"]

        # define the path of different data source,
        #   the annotation is .xml, the sentence is in .txt
        self.source_data_types = ["Images"]
        self.source_data_file_formats = ["jpg"]
        self.build_source_data_structure()

        # Obtain which split to use:
        #  refclef, refcoco, refcoco+ and refcocog
        self.sub_dataset = Config().data.sub_dataset
        # Obtain which specific setting to use:
        #  unc, google
        self.split_by_name = Config().data.split_by
        if self.sub_dataset not in self.sub_datasets_name:
            logging.info(
                "%s does not exsit in the official datasets %s",
                self.sub_dataset,
                self.sub_datasets_name,
            )
        # download the public official code and the required config
        base_data_path = self.data_info["base_data_path"]
        source_images_dir_path = self.data_info["Images"]["path"]
        download_subdataset_url = (
            Config().data.sub_datasets_download_url + self.sub_dataset + ".zip"
        )
        data_utils.download_url_data(
            download_url_address=download_subdataset_url,
            put_data_dir=base_data_path,
        )

        self._dataset_refer = refer_api.REFER(
            data_root=base_data_path,
            image_dataroot=source_images_dir_path,
            dataset=self.sub_dataset,
            splitBy=self.split_by_name,
        )
        self.meta_datacatalog_filename = (
            f"vggbase_{self.sub_dataset}_{self.split_by_name}_metadata"
        )
        self.datacatalog_filename = (
            f"vggbase_{self.sub_dataset}_{self.split_by_name}_datacatalog"
        )

        self.build_splits_structure()

        # generate the splits information txt for further utilization
        logging.info("Integrating VGGBase datacatalog...")
        refer_utils.integrate_data(
            splits_info=self.splits_info,
            dataset_refer_op=self._dataset_refer,
            metadata_filepath=self.get_metadatacatalog_path(),
            datacatalog_filename_fn=self.get_split_datacatalog_path,
        )
        self.compute_splits_data_statistics()

    def build_splits_structure(self):
        # generate path/type information for splits
        # refer it game should create the file of different splits
        base_data_path = self.data_info["base_data_path"]
        base_filename = f"{self.sub_dataset}_{self.split_by_name}"
        for split_type in self.splits:
            split_filename = f"{split_type}_{base_filename}"
            self.splits_info[split_type]["path"] = self.source_data_dir_path
            split_filepath = os.path.join(base_data_path, split_filename + ".json")
            self.splits_info[split_type]["split_file"] = split_filepath

            if not data_utils.is_data_exist(split_filepath):
                logging.info("Creating split file %s", split_filepath)
                split_ref_ids = self._dataset_refer.getRefIds(split=split_type)
                with open(split_filepath, "w", encoding="utf-8") as f:
                    json.dump(split_ref_ids, f)
            else:
                logging.info("Existed split file %s", split_filepath)

    def get_phase_dataset(self, phase: str, image_transform, text_transform):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
        phase_datacatalog_filepath = self.get_split_datacatalog_path(phase)
        dataset = ReferItGameDataset(
            datacatalog_filepath=phase_datacatalog_filepath,
            image_transform=image_transform,
            text_transform=text_transform,
        )
        return dataset
