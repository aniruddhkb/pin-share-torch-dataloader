"""Testing inference."""

from torch.nn import Module
from torchvision import models  # type: ignore


def get_model() -> Module:
    """Get the model."""
    return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)


# Profiling config
COMPILER_ACTIVE = True
BATCH_SIZE = 4
DLOADER_NUM_WRKRS = 6  # 9 or below at 1024, possibly at 512.
N_REPEATS = 1  # 4 at 512, 1024 bs, 2 below.

PROFILER_ACTIVE = True
TORCH_PROFILE_WORKERS = False

PROFILER_MODE = "nsys"
WAIT_BATCHES = 15000  # 13000, 7000, 3200, 1650, 900, 450, 450, 250
WARMUP_BATCHES = 1000
RECORD_BATCHES = 200  # 300, 300, 300, 150, 75, 75, 50

TORCH_PROFILER_OUT_DIR = (
    f"./torch_profiler/expt_bs{BATCH_SIZE}_ca{COMPILER_ACTIVE}_dnw{DLOADER_NUM_WRKRS}_withworkers"
)

# ------------------------------------------------------------


import torch  # noqa: E402
from torch.multiprocessing import set_start_method  # noqa: E402

if __name__ == "__main__":
    set_start_method("spawn", force=True)  # type: ignore


import time  # noqa: E402  # noqa: E402
import warnings  # noqa: E402

from tqdm import tqdm  # type:ignore  # noqa: E402

from async_load.decoded_to_file.cached_vis_dset import CachedVisionDataset  # noqa: E402
from async_load.loader import CudaDataLoader  # noqa: E402
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


def get_dloader() -> CudaDataLoader:
    """Get the data loaders for training and validation."""
    main_train_dset = CachedVisionDataset(cache_dir=f"/dsets/imagenet-1k-{_IMG_SIZE_STR}_fp16/{_SPLIT}/")
    train_dset = RepeatedDataset(main_train_dset, N_REPEATS)
    return CudaDataLoader(
        base_dset=train_dset,
        batch_size=BATCH_SIZE,
        n_wrkrs=DLOADER_NUM_WRKRS,
        shuffle=True,
        n_cubufs=2,
        profile_trace=TORCH_PROFILE_WORKERS,
        profile_trace_path=TORCH_PROFILER_OUT_DIR,
        profile_wait=WAIT_BATCHES // DLOADER_NUM_WRKRS,
        profile_warmup=WARMUP_BATCHES // DLOADER_NUM_WRKRS,
        profile_record=RECORD_BATCHES // DLOADER_NUM_WRKRS,
    )


def main() -> None:
    """Start main process."""
    x: torch.Tensor
    y: torch.Tensor

    try:
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
                for ctx in tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=2.0):
                    with ctx:
                        prof.nextstep()
                        x, y = ctx.buffers
                        with torch.inference_mode():
                            yhats_logits = model(x)
                        metric(yhats_logits, y)
                t1 = time.time()
                print("ACCURACY:", metric.compute())
                print(f"SAMPLE THROUGHPUT: {len(train_loader) * BATCH_SIZE / (t1 - t0):.3f} samples/s")
    except (KeyboardInterrupt, TorchProfilerCompleteError):
        pass


if __name__ == "__main__":
    main()
