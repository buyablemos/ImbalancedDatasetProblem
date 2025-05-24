import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class CapsuleDataset(Dataset):
    def __init__(self,
                 pos_dir: str,
                 neg_dirs,
                 transform=None,
                 max_per_dir: int = None):
        """
        neg_dirs: either a single path string or a list of paths
        max_per_dir: if set, take at most this many files from EACH directory
        """
        if isinstance(neg_dirs, str):
            neg_dirs = [neg_dirs]

        self.transform = transform
        self.samples = []

        def _gather_files(directory):
            files = sorted(glob.glob(os.path.join(directory, "*")))
            if max_per_dir is not None:
                return files[:max_per_dir]
            return files

        if pos_dir is not None:
            pos_files = sorted(glob.glob(os.path.join(pos_dir, "*")))
            self.samples += [(p, 0) for p in pos_files]

        # negatives (label=1)
        for nd in neg_dirs:
            neg_files = _gather_files(nd)
            self.samples += [(p, 1) for p in neg_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label



class NegativeOnlySubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.samples = [i for i in indices if base_dataset.samples[i][1] == 1]
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, _= self.base_dataset[self.samples[idx]]
        return img