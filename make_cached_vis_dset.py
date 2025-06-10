"""Make a cached dataset for the smaller dsets."""

import torch
from datasets import load_dataset  # type: ignore
from torchvision.transforms import v2 as T  # type: ignore  # noqa: N812
from tqdm import tqdm

from async_load.decoded_to_file.cached_vis_dset import CachedVisionDataset
from async_load.utils._hf_img_dset_to_pt import HFImageTorchDset

DSET_NAME, SPLIT = "benjamin-paine/imagenet-1k-32x32", "validation"
DSET_DIR = f"/dsets/imagenet-1k-32x32_fp16/{SPLIT}"

DTYPE = torch.float16

MEANS = [0.4848, 0.4579, 0.4078]
STD_DEVS = [0.2669, 0.2606, 0.2732]

TRANSFORM = T.Compose(
    [
        T.ToDtype(dtype=DTYPE, scale=True),  # type: ignore
        T.Normalize(mean=MEANS, std=STD_DEVS),
    ],
)

hf_dset = load_dataset(DSET_NAME, split=SPLIT)
dset = HFImageTorchDset(hf_dset=hf_dset, transform=TRANSFORM)  # type: ignore

CachedVisionDataset.make_cached_dset(dset, cache_dir=DSET_DIR)

cached_dset = CachedVisionDataset(cache_dir=DSET_DIR)
len_ = len(cached_dset)

for i in tqdm(range(len_)):
    cached_tens, cached_label = cached_dset[i]
    tens, label = dset[i]
    assert torch.equal(tens, cached_tens)
    assert label == cached_label
print("Cached dataset is valid.")
