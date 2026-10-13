# SuperMapNet (arXiv 2505.13856) -- https://github.com/zhouruqin/SuperMapNet
#
# Built on the PivotNet/BeMapNet ("mapmaster") codebase, so this image also
# runs the PivotNet and BeMapNet configs in configs/.
#
# Diverges from README.md's install steps (conda py3.8, torch 1.10.1+cu111,
# mmcv 1.7.1, detectron2 for cu111/torch1.10): that stack predates CUDA 11.8,
# the first CUDA release with real Hopper (H100, sm_90) support. Bumped to
# torch 2.1.0/CUDA 11.8 + mmcv-full 1.7.2, following the same fix already
# applied to the sibling MapTRv2, GeMap, MapDiffusion and PseudoMapTrainer
# codebases' Dockerfiles.
#
# --- Build / run ---
#   docker build -t supermapnet .
#   docker run --gpus all -it --shm-size=16g \
#     -v /path/to/nuscenes:/workspace/SuperMapNet/data/nuscenes \
#     -v /path/to/argoverse2:/workspace/SuperMapNet/data/argoverse2 \
#     -v /path/to/weights:/workspace/SuperMapNet/assets/weights \
#     -v /path/to/outputs:/workspace/SuperMapNet/outputs \
#     supermapnet
#
# The configs use relative paths (DATA_ROOT = './data/nuscenes/', split_dir =
# 'assets/splits/nuscenes'), and run.sh writes checkpoints under ./outputs, so
# mount each of those inside the workdir. --shm-size matters: the default
# 64 MB is not enough for the dataloader workers these configs use.
ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 8.6 covers Ampere (e.g. RTX 3070/3090); 9.0+PTX covers Hopper (H100) and,
# via PTX forward-compat JIT, anything newer released after this image (e.g.
# Blackwell RTX 50-series).
ENV TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

# libgl1/libusb-1.0-0 are here only for Open3D, which the nuScenes and
# Argoverse 2 datasets import at module scope for their LiDAR voxel
# down-sampling (mapmaster/dataset/nuscenes_pivotnet.py, av2_dataset.py) and
# which links libGL unconditionally -- unlike the sibling images, this one
# cannot be made completely GL-free. OpenCV is still swapped to the headless
# build at the end, so cv2 at least never adds to that exposure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libusb-1.0-0 \
        libxext6 \
        libxrender-dev \
        nano \
        ninja-build \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# mmcv, bumped from README.md's 1.7.1 to 1.7.2 (the version the sibling images
# are proven on; a patch release on the same torch-2.x build-fix line). This
# repo only reaches for mmcv's pure-Python layers -- mmcv.cnn's conv/norm
# builders in the backbones, mmcv.runner/mmcv.fileio in the Swin checkpoint
# loader, mmcv.parallel.DataContainer in the AV2 pipelines -- and none of
# mmcv.ops: every CUDA kernel this repo executes outside torch's own comes
# from the in-tree deformable-attention op built further down. mmcv-full is
# installed rather than lite mmcv anyway, and from source rather than from
# OpenMMLab's prebuilt wheel index, so that if any mmcv.ops path is ever hit
# it is built for this image's own TORCH_CUDA_ARCH_LIST. OpenMMLab's prebuilt
# cu118/torch2.1.0 wheel was compiled with their own (unknown,
# non-configurable) arch list, which does not include Hopper (sm_90) and has
# no +PTX fallback baked in -- invisible on Ampere, and surfacing on an actual
# H100 only as `RuntimeError: CUDA error: no kernel image is available for
# execution on the device` the first time such an op runs.
#
# Note there is deliberately no mmdet/mmsegmentation/mmdet3d here, unlike the
# sibling MapTR/GeMap/MapDiffusion images: this codebase is not an mmdet
# plugin -- it has its own trainer (mapmaster/engine) driving torch DDP
# directly, so mmcv's MMDataParallel/MMDistributedDataParallel (and the
# torch-2.x patches the sibling images apply to them) are never involved.
RUN MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"

# detectron2, for PointRend's point_sample/get_uncertain_point_coords_with_
# randomness, which the mask and recovery losses use
# (mapmaster/models/utils/{mask,recovery}_loss.py). README.md installs the
# cu111/torch1.10 prebuilt wheel; detectron2 has no release supporting torch
# 2.x (v0.6 is from 2021), so it is built from the same pinned main-branch
# commit the sibling PseudoMapTrainer image uses. Building from source is
# right here anyway -- the prebuilt index only ever covered old torch/CUDA
# pairs, and detectron2's CUDA ops need this image's arch list like everything
# else. Its setup.py maps projects/PointRend into detectron2.projects, so a
# non-editable install still exposes point_rend.
RUN pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"

# torch-scatter, behind the pillar/voxel encoders' scatter_mean/scatter_sum
# (mapmaster/dataset/voxel.py, models/lidar_encoder/pointpillar.py). Built
# from source rather than taken from PyG's prebuilt wheel index for the same
# reason as mmcv-full: those wheels' baked-in arch list is not ours and cannot
# be assumed to carry sm_90 or PTX. --no-build-isolation because its setup.py
# imports torch at build time and does not declare it as a build requirement;
# MAX_JOBS caps peak build memory.
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation "torch-scatter==2.1.2"

# inplace-abn, for the InPlaceABNSync used by the cross-view attention and BEV
# fusion encoders (mapmaster/models/{cross_encoder/Cross_Attn,fusionbev_
# encoder/BEVFusion}.py -- both imported at module scope by
# mapmaster/models/network.py even in the configs that do not enable them).
# 1.1.0 is the only release on PyPI and ships no wheels, so this is a source
# build: setuptools_scm goes in first because inplace-abn's setup.py asks for
# it through the legacy setup_requires, which modern setuptools no longer
# fetches on its own, and --no-build-isolation because its setup.py imports
# torch. Its CUDAExtension is selected by torch.has_cuda, a build-time
# property of the torch build rather than a runtime probe, so it compiles on
# this GPU-less builder.
RUN pip install --no-cache-dir setuptools_scm \
    && pip install --no-cache-dir --no-build-isolation "inplace-abn==1.1.0"

RUN conda clean --all

WORKDIR /workspace/SuperMapNet
COPY . .

# The in-tree Deformable-DETR multi-scale deformable-attention op
# (README.md step f, `bash make.sh`), patched for CUDA 11.8 / torch 2.1:
#
#   * setup.py hard-codes `-arch=sm_60` plus gencodes up to sm_75 in its nvcc
#     flags. torch's cpp_extension skips its own TORCH_CUDA_ARCH_LIST handling
#     entirely as soon as it sees an 'arch' flag in extra_compile_args, so
#     those five lines -- not this Dockerfile's arch list -- would decide what
#     the op is compiled for: no sm_86, no sm_90, no PTX. That is invisible
#     until the first ms_deform_attn call on an Ampere or Hopper card, which
#     then fails with `RuntimeError: CUDA error: no kernel image is available
#     for execution on the device`. Dropping them hands the decision back to
#     TORCH_CUDA_ARCH_LIST ("8.6 9.0+PTX", set above).
#   * setup.py gates the CUDA extension on torch.cuda.is_available() and
#     raises NotImplementedError otherwise -- `docker build` has no GPU, so
#     the guard also honours FORCE_CUDA here, as every other CUDA op in these
#     images already does.
#   * THC/THCAtomics.cuh (for gpuAtomicAdd) was removed from ATen after torch
#     1.10; ATen/cuda/Atomic.cuh is its replacement. Same fix the sibling
#     images apply to their vendored mmdetection3d.
RUN python - <<'PY'
import pathlib

setup = pathlib.Path('mapmaster/models/bev_decoder/deform_transformer/ops/setup.py')
src = setup.read_text()
old_guard = "    if torch.cuda.is_available() and CUDA_HOME is not None:"
new_guard = ('    if (torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1") '
             'and CUDA_HOME is not None:')
assert old_guard in src, 'deform_transformer setup.py not in the expected shape'
src = src.replace(old_guard, new_guard)
for arch in ['            "-arch=sm_60",\n',
             '            "-gencode=arch=compute_60,code=sm_60",\n',
             '            "-gencode=arch=compute_61,code=sm_61",\n',
             '            "-gencode=arch=compute_70,code=sm_70",\n',
             '            "-gencode=arch=compute_75,code=sm_75",\n']:
    assert arch in src, arch
    src = src.replace(arch, '')
setup.write_text(src)

cuh = pathlib.Path('mapmaster/models/bev_decoder/deform_transformer/ops/src/cuda/'
                   'ms_deform_im2col_cuda.cuh')
s = cuh.read_text()
assert '#include <THC/THCAtomics.cuh>' in s, 'ms_deform_im2col_cuda.cuh not in the expected shape'
cuh.write_text(s.replace('#include <THC/THCAtomics.cuh>', '#include <ATen/cuda/Atomic.cuh>'))
PY
RUN cd mapmaster/models/bev_decoder/deform_transformer/ops && python setup.py build install

# requirement.txt, with three defects fixed on the way in -- as committed it
# cannot be handed to pip at all:
#   * `torch_scatter=2.1.2` uses a single '=' (not a valid requirement
#     specifier) and is a source build needing FORCE_CUDA/--no-build-isolation
#     -- installed above instead, so it is dropped here.
#   * `inplace_abn` likewise needs its own build handling -- installed above.
#   * `iceream` is a typo for `icecream`, which several modules import.
# The rest is installed as the repo asks. scikit-image is pinned to 0.19.3,
# the first release of that series with cp310 wheels, and numba to 0.58.1, the
# first with cp310 wheels that still accepts the numpy pinned below.
RUN sed -e '/^torch_scatter=/d' \
        -e '/^inplace_abn$/d' \
        -e 's/^iceream$/icecream/' \
        -e 's/^scikit-image$/scikit-image==0.19.3/' \
        -e 's/^numba$/numba==0.58.1/' \
        requirement.txt > /tmp/requirement.txt \
    && pip install --no-cache-dir -r /tmp/requirement.txt

# Dependencies this repo imports but never lists: Open3D (LiDAR down-sampling
# in the dataset classes), av2 (the Argoverse 2 datasets and map extractors),
# geopandas (av2map_extractor_pmapnet.py). av2 0.2.1 is the version the
# sibling images pin; geopandas is held at 0.13.2, the last series built
# against shapely 1.x, so it does not drag in shapely 2 (see below).
RUN pip install --no-cache-dir \
        "av2==0.2.1" \
        "geopandas==0.13.2" \
        "open3d==0.18.0"

# nuscenes-devkit 1.1.9 (as requirement.txt pins) does
# `plt.style.use('seaborn-whitegrid')` at module scope in
# nuscenes/map_expansion/map_api.py, which this repo imports. That style name
# was removed in matplotlib 3.8, so an unpinned matplotlib turns any nuScenes
# run into an `OSError: 'seaborn-whitegrid' is not a valid package style` at
# import time. 3.7.5 is the last release that still has it.
RUN pip install --no-cache-dir "matplotlib==3.7.5"

# av2 imports numpy.typing (added in numpy 1.20) at module load time. Pin
# below 1.24 to keep the np.float/np.int aliases mmcv-full may still use
# (removed in numpy 1.24), and to stay inside numba 0.58.1's numpy bound.
RUN pip install --no-cache-dir "numpy==1.23.5"

# ...which makes the pyarrow that av2 pulls in unusable. av2 requires pyarrow
# with no version bound, so it resolves to the latest, whose wheel is built
# against the numpy 2 C-API and declares no numpy requirement of its own -- so
# pip reports no conflict, and the mismatch only shows up at import time as a
# bare SIGSEGV with no Python traceback (reached via pandas.compat.pyarrow).
# 12.0.1 is a numpy-1.x-era release and satisfies av2's unbounded requirement.
RUN pip install --no-cache-dir "pyarrow==12.0.1"

# requirement.txt pins shapely==1.8.5, but nuscenes-devkit (unpinned shapely
# dependency) silently upgrades it to 2.x. shapely 2.0 changed STRtree.query()
# to return integer indices instead of geometry objects and dropped the
# iteration protocol on multi-part geometries -- breaking changes the map
# extractors and the chamfer-distance evaluation rely on. Re-pin as the last
# install so it sticks (1.8.5.post1 is 1.8.5 plus wheel fixes only).
RUN pip install --no-cache-dir "shapely==1.8.5.post1"

# nuscenes-devkit/mmcv-full/av2 all transitively pull in non-headless
# opencv-python, which links its GUI backend against libGL.so.1. On
# Singularity/Apptainer clusters with --nv, the host's newer libGL.so.1 gets
# bind-mounted over the container's own and fails to load against this image's
# older glibc (GLIBC_ABI_DT_RELR, added in glibc 2.36, missing here). Swap to
# the headless build -- same OpenCV, no GL/X11 linkage at all -- so cv2 never
# touches libGL.so.1 regardless of what the host injects. requirement.txt asks
# for opencv-contrib-python, so the contrib build is kept, headless.
RUN OPENCV_VERSION=$(pip show opencv-contrib-python | sed -n 's/^Version: //p') \
    && pip uninstall -y opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless \
    && pip install --no-cache-dir \
        "opencv-contrib-python-headless==${OPENCV_VERSION}"

# The SAM2 backbone (configs/pivotnet_nuscenes_sam2.py,
# mapmaster/models/backbone/sam2feature/) is deliberately NOT installed:
# facebookresearch/sam2 requires torch>=2.3.1, which is incompatible with
# the mmcv-full 1.x stack this image is built on. The repo's own
# sam2feature/model.py guards its `import sam2` for exactly this case, so
# the other configs (PivotNet/BeMapNet with ResNet, EfficientNet-B0 and
# Swin-T backbones) import fine and constructing a SAM2Backbone here raises
# ImportError naming the reason.
#
# (Unrelated to this image: configs/bemapnet_nuscenes_effb0.py imports a
# `BeMapNet` class that mapmaster/models/network.py does not define -- it
# only has `MapMaster` -- so that one config is broken upstream. The other
# seven configs import cleanly here.)

# Datasets (./data), backbone init weights (./assets/weights, README.md
# section "Weights: Public-Pretrain-Models") and training output (./outputs,
# where run.sh looks for checkpoints) are expected to be mounted at runtime.
RUN mkdir -p data assets/weights outputs

# run.sh does `export PYTHONPATH=$(pwd)`, but set it here too so `python
# configs/....py` works from any shell in the container.
ENV PYTHONPATH=/workspace/SuperMapNet

CMD ["/bin/bash"]
