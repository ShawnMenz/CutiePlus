import sys
from typing import Literal, Optional

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from gui.resource_manager import ResourceManager


class PropagationReader(Dataset):
    """
    Yields (display image, display tensor, model tensor). The model always sees the default
    (anchor) modality; the display follows display_modality when a task has a second one.
    """
    def __init__(self,
                 res_man: ResourceManager,
                 start_ti: int,
                 direction: Literal['forward', 'backward'],
                 display_modality: Optional[str] = None):
        self.res_man = res_man
        self.start_ti = start_ti
        self.direction = direction
        self.display_modality = display_modality

        # skip the first frame
        if self.direction == 'forward':
            self.start_ti += 1
            self.length = self.res_man.T - self.start_ti
        elif self.direction == 'backward':
            self.start_ti -= 1
            self.length = self.start_ti + 1
        else:
            raise NotImplementedError

        self.to_tensor = ToTensor()

    def __getitem__(self, index: int):
        if self.direction == 'forward':
            ti = self.start_ti + index
        elif self.direction == 'backward':
            ti = self.start_ti - index
        else:
            raise NotImplementedError

        assert 0 <= ti < self.res_man.T

        image = self.res_man.get_image(ti)
        image_torch = self.to_tensor(image)
        if self.display_modality in (None, self.res_man.default_modality):
            return image, image_torch, image_torch

        display_image = self.res_man.get_image(ti, self.display_modality)
        return display_image, self.to_tensor(display_image), image_torch

    def __len__(self):
        return self.length


def get_data_loader(dataset: Dataset, num_workers: int):
    if 'linux' in sys.platform:
        loader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=lambda x: x)
    else:
        print(f'Non-linux platform {sys.platform} detected, using single-threaded dataloader')
        loader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x: x)
    return loader