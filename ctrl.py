"""Testing inference."""

from torch.nn import Module
from torchvision import models  # type: ignore


def get_model() -> Module:
    """Get the model."""
    # return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)


# Profiling config
COMPILER_ACTIVE = True
BATCH_SIZE = 16
DLOADER_NUM_WRKRS = 4  # 9 or below at 1024, possibly at 512.
N_REPEATS = 2  # 4 at 512, 1024 bs, 2 below.

PROFILER_ACTIVE = True
TORCH_PROFILE_WORKERS = True

PROFILER_MODE = "torch"
WAIT_BATCHES = 1  # 13000, 7000, 3200, 1650, 900, 450, 450, 250
WARMUP_BATCHES = 1
RECORD_BATCHES = 200  # 300, 300, 300, 150, 75, 75, 50

TORCH_PROFILER_OUT_DIR = (
    f"./torch_profiler/ctrl_bs{BATCH_SIZE}_ca{COMPILER_ACTIVE}_dnw{DLOADER_NUM_WRKRS}_withworkers"
)

# Dataset config
# ------------------------------------------------------------


import torch  # noqa: E402
from torch.multiprocessing import set_start_method  # noqa: E402

if __name__ == "__main__":
    set_start_method("spawn", force=True)  # type: ignore


import time  # noqa: E402
import warnings  # noqa: E402

from torch.utils import data  # noqa: E402
from tqdm import tqdm  # type:ignore  # noqa: E402

from async_load.decoded_to_file.cached_vis_dset import CachedVisionDataset  # noqa: E402
from async_load.utils._device_accuracy import CategoricalAccuracy  # noqa: E402
from async_load.utils._prflrs import TorchProfilerCompleteError, get_profiler  # noqa: E402
from async_load.utils._repeated_dset import RepeatedDataset  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


_IMG_SIZE = 224
_N_EPOCHS = 100
_SPLIT = "validation"
_DEVICE = torch.device("cuda")
_IMG_SIZE_STR = f"{_IMG_SIZE}x{_IMG_SIZE}"
_MODEL_DTYPE = torch.float16


if TORCH_PROFILE_WORKERS and (PROFILER_MODE != "torch" or not PROFILER_ACTIVE):
    msg = "Torch profiler workers only supported in torch profiler mode."
    raise ValueError(msg)

if TORCH_PROFILE_WORKERS and __name__ != "__main__":
    import atexit

    profiler_obj = get_profiler(
        "torch_no_cuda",
        0,
        0,
        1,
        TORCH_PROFILER_OUT_DIR,
        bypassed=not PROFILER_ACTIVE,
    )
    profiler_obj.__enter__()
    atexit.register(profiler_obj.nextstep)


def get_dloader() -> data.DataLoader:
    """Get the data loaders for training and validation."""
    main_train_dset = CachedVisionDataset(cache_dir=f"/dsets/imagenet-1k-{_IMG_SIZE_STR}_fp16/{_SPLIT}/")
    main_train_dset = CachedVisionDataset(cache_dir=f"/dsets/imagenet-1k-{_IMG_SIZE_STR}_fp16/{_SPLIT}/")
    train_dset = RepeatedDataset(main_train_dset, N_REPEATS)
    train_loader: data.DataLoader = data.DataLoader(
        train_dset,
        batch_size=BATCH_SIZE,
        num_workers=DLOADER_NUM_WRKRS,
        pin_memory=True,
        persistent_workers=DLOADER_NUM_WRKRS > 0,
        shuffle=True,
        drop_last=True,
    )
    return train_loader


def main() -> None:
    """Start main process."""
    x: torch.Tensor
    y: torch.Tensor
    x_cpu: torch.Tensor
    y_cpu: torch.Tensor

    try:
        first_time = True
        with (
            get_profiler(
                PROFILER_MODE,
                WAIT_BATCHES,
                WARMUP_BATCHES,
                RECORD_BATCHES,
                TORCH_PROFILER_OUT_DIR,
                bypassed=not PROFILER_ACTIVE,
            ) as prof,
            torch.no_grad(),
            torch.inference_mode(),
        ):
            model = get_model()
            model = model.eval()
            model.to(dtype=_MODEL_DTYPE, device=_DEVICE)

            model.compile(
                fullgraph=True,
                dynamic=False,
                mode="max-autotune",
                backend="inductor",
                disable=not COMPILER_ACTIVE,
            )

            model = get_model()
            model = model.eval()
            model.to(dtype=_MODEL_DTYPE, device=_DEVICE)

            model.compile(
                fullgraph=True,
                dynamic=False,
                mode="max-autotune",
                backend="inductor",
                disable=not COMPILER_ACTIVE,
            )

            train_loader = get_dloader()
            metric = CategoricalAccuracy(
                n_classes=1000,
                batch_size=BATCH_SIZE,
                max_n_batches=len(train_loader),
                dtype=_MODEL_DTYPE,
            )
            for epoch in range(_N_EPOCHS):
                metric.reset()
                t0 = time.time()
                for x_cpu, y_cpu in tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=2.0):
                    prof.nextstep()
                    if first_time:
                        x, y = (
                            x_cpu.to(device=_DEVICE, dtype=_MODEL_DTYPE),
                            y_cpu.to(device=_DEVICE),
                        )
                        print(x_cpu.dtype, x.dtype)
                        first_time = False
                    else:
                        x.copy_(x_cpu, non_blocking=True)
                        y.copy_(y_cpu, non_blocking=True)

                    yhats_logits = model(x)
                    metric(yhats_logits, y)
                t1 = time.time()
                print("ACCURACY:", metric.compute())
                print(f"SAMPLE THROUGHPUT: {len(train_loader) * BATCH_SIZE / (t1 - t0):.3f} samples/s")
    except (KeyboardInterrupt, TorchProfilerCompleteError):
        pass
    finally:
        del train_loader


if __name__ == "__main__":
    main()
