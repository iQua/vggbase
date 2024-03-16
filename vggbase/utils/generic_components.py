"""
Implementation of generic components that maintain consistent across the framework.
"""

from dataclasses import dataclass, fields
from collections import UserList

import torch
from transformers.utils import ModelOutput


class BaseVGList(UserList):
    """Basic structure of `List` used in VGGBase."""

    def to(self, device):
        """Assign the tensor item into the specific device."""
        for sample_idx, sample_target in enumerate(self.data):
            if hasattr(sample_target, "to"):
                if isinstance(sample_target, torch.Tensor):
                    sample_target = sample_target.to(device)
                else:
                    sample_target.to(device)

                super().__setitem__(sample_idx, sample_target)

    @property
    def device(self):
        """Get the device of the inner tensors."""
        return tuple(
            [
                (sample_idx, sample_target.device)
                for sample_idx, sample_target in enumerate(self.data)
                if hasattr(sample_target, "device")
            ]
        )


@dataclass
class FieldFrozenContainer(ModelOutput):
    """A Container whose fields are frozen."""

    def __setattr__(self, name, value):
        class_fields_name = [class_field.name for class_field in fields(self)]

        if name not in class_fields_name:
            raise KeyError(f"{name} does not exist in fields {class_fields_name}.")
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        class_fields_name = [class_field.name for class_field in fields(self)]

        if key not in class_fields_name:
            raise KeyError(f"{key} does not exist in fields {class_fields_name}.")
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to(self, device):
        """Assign inner tensors to the device."""
        class_fields = fields(self)
        for field in class_fields:
            field_v = getattr(self, field.name)

            if hasattr(field_v, "to"):
                if isinstance(field_v, torch.Tensor):
                    field_v = field_v.to(device)
                else:
                    field_v.to(device)

                super().__setitem__(field.name, field_v)

    @property
    def device(self):
        """Get the device of the inner tensors."""
        collected_devices = []
        class_fields = fields(self)

        for field in class_fields:
            field_v = getattr(self, field.name)
            if hasattr(field_v, "device"):
                collected_devices.append((field.name, field_v.device))

        return tuple(collected_devices)

    @staticmethod
    def get_json(container: ModelOutput):
        """Get the data that can be saved with the json type."""
        data_dict = {}
        for field in fields(container):
            field_v = getattr(container, field.name)
            if torch.is_tensor(field_v):
                field_v = field_v.cpu().detach().numpy().tolist()
            if isinstance(field_v, ModelOutput):
                field_v = FieldFrozenContainer.get_json(field_v)
            data_dict[field.name] = field_v

        return data_dict
