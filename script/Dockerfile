# syntax=docker/dockerfile:1.6
#
# The base image of FuncBind's images (voxbind:allinone): it keeps the voxdock and moleval envs that
# FuncBind's MCP docking calls by path. The pipeline image is env/Dockerfile (voxbind + voxeval).
#
# VoxBind — ALL-IN-ONE image for the full density-conditioned pipeline:
#   data preprocess -> train -> sample (GPU)  +  Vina docking + PoseCheck eval (CPU).
#
# It carries THREE conda envs (the eval stack pins conflict with the GPU stack,
# so they cannot share one interpreter) plus a TargetDiff sibling clone:
#   voxbind  torch cu118 + pyuul + hydra + rdkit + openbabel + gemmi   (train/sample)
#   voxdock  python 3.8  + vina 1.2.2 + meeko 0.1.dev3 + pdb2pqr        (docking)
#   moleval  python 3.10 + posecheck 1.3.1 + prolif + posebusters      (pose quality)
#
# CUDA runtime libs come from pytorch-cuda inside the voxbind env, so the host
# only needs an NVIDIA driver new enough for CUDA 11.8 (>= 520) + the NVIDIA
# Container Toolkit. Run with `--gpus all`. The image is large (~15-20 GB); the
# eval envs add ~3 GB. Build from the REPO ROOT with this file:
#
#   docker build -f script/Dockerfile -t voxbind:allinone .
#
#   docker run --rm -it --gpus all --shm-size=16g \
#     -v $PWD/dataset/data:/workspace/VoxBind/dataset/data \
#     -v $PWD/model_zoo:/workspace/VoxBind/model_zoo \
#     -v $PWD/task2-ligand_generation/exps:/workspace/VoxBind/task2-ligand_generation/exps \
#     voxbind:allinone
#   # then inside:  bash script/01_download_data.sh && bash script/02_preprocess_data.sh && ...
#
# Binding affinity in one command (pretraining, features, the six test sets; configs in
# task1-affinity_regression/configs/), as your own user so the mounted outputs stay yours:
#   docker run --rm --gpus all --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD/dataset/data:/workspace/VoxBind/dataset/data \
#     -v $PWD/pretrain/exps:/workspace/VoxBind/pretrain/exps \
#     -v $PWD/pretrain/log:/workspace/VoxBind/pretrain/log \
#     -v $PWD/results:/workspace/VoxBind/results \
#     voxbind:allinone /opt/conda/envs/voxbind/bin/python task1-affinity_regression/run.py \
#     task1-affinity_regression/configs/ab7v2p1_clean.yaml
# GPU ids in a config count the GPUs the container sees. W&B is off in the image; pass
# -e WANDB_MODE=online -e WANDB_API_KEY=... to log a run.
#
# Data + weights are NOT baked in (they are large and licence-encumbered); pull
# them at runtime with script/01_download_data.sh, or mount them as above.

ARG MICROMAMBA_VERSION=1.5.10
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates curl unzip \
        libxrender1 libxext6 libsm6 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /workspace/VoxBind

# --- env 1: voxbind (GPU pipeline + Vina docking). Build from the lock (vina 1.2.2)
#     when present, else env.yaml. Copy just the specs first so this heavy layer
#     caches across code edits. This env alone runs docking + chem/geometry eval. ---
COPY --chown=$MAMBA_USER:$MAMBA_USER env/ /tmp/env/
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN if [ -f /tmp/env/voxbind.conda-linux-64.lock ]; then \
        micromamba create -y -n voxbind --file /tmp/env/voxbind.conda-linux-64.lock && \
        micromamba run -n voxbind pip install --no-cache-dir -r /tmp/env/voxbind.pip.lock.txt ; \
    else \
        micromamba env create -y -f /tmp/env.yaml ; \
    fi \
 && micromamba clean --all --yes

# --- env 2: voxdock (Vina 1.2.2 docking stack; pins are load-bearing) ---
RUN micromamba create -y -n voxdock -c conda-forge \
        python=3.8.16 numpy=1.24.3 scipy=1.10.1 rdkit=2022.03.2 openbabel=3.1.1 \
        pdb2pqr easydict pip \
 && micromamba run -n voxdock pip install --no-cache-dir \
        vina==1.2.2 meeko==0.1.dev3 pdb2pqr==3.6.1 propka==3.5.0 \
        mmcif-pdbx==2.0.1 docutils==0.17.1 \
        "git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git@aee55d50d5bdcfdbcd80220499df8cde2a8f4b2a" \
 && micromamba clean --all --yes

# --- env 3: moleval (PoseCheck 1.3.1 + PoseBusters + ProLIF) ---
RUN micromamba create -y -n moleval -c conda-forge python=3.10 reduce pip \
 && micromamba run -n moleval pip install --no-cache-dir \
        posebusters 'pandas>=2.2.3' prolif datamol hydride biopython rdkit \
 && micromamba run -n moleval pip install --no-cache-dir --no-deps 'posecheck==1.3.1' \
 && micromamba clean --all --yes

# --- TargetDiff: sibling of the repo (VOXBIND.parent/targetdiff).
#     Its data/test_set/ (full receptors) is a separate download — pull via
#     script/01_download_data.sh or mount it. ---
RUN git clone --depth 1 https://github.com/guanjq/targetdiff /workspace/targetdiff

# --- project source + editable install into the voxbind env ---
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/VoxBind
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba run -n voxbind pip install --no-cache-dir -e .

ENV HOME=/home/$MAMBA_USER \
    VOXBIND_PY=/opt/conda/envs/voxbind/bin \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    CONDA_LAUNCHER=micromamba \
    REPO_ROOT=/workspace/VoxBind \
    TARGETDIFF_ROOT=/workspace/targetdiff \
    FULL_RECEPTOR_ROOT=/workspace/targetdiff/data/test_set/test_set

CMD ["bash"]
