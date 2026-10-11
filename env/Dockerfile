# syntax=docker/dockerfile:1.6
#
# CoDE image: the two environments of the pipeline, restored from their locks.
#   voxbind  (py3.10)  training: pretraining, affinity probes, ligand generation (env/voxbind.conda-lock.yml)
#   voxeval  (py3.10)  evaluation: QED/SA/diversity, AutoDock Vina 1.2.2, PoseCheck 1.3.1 and
#                      PoseBusters 0.6.5 (env/voxeval.* locks, built by script/00_setup_env.sh voxeval)
#
# pytorch-cuda 11.8 comes from the lock, so the host only needs an NVIDIA driver (>= 520 for CUDA 11.8)
# and the NVIDIA Container Toolkit. Data, weights and outputs are mounted at run time, not baked in.
#
# BUILD FROM THE REPO ROOT (context = repo root; .dockerignore keeps data/weights out):
#   docker build -f env/Dockerfile -t voxbind:env .
#
# Run as your own user so the mounted outputs stay yours. GPU ids in a config count the GPUs the
# container sees (--gpus '"device=4,5,6,7"' -> 0-3). W&B is off in the image; pass
# -e WANDB_MODE=online -e WANDB_API_KEY=... to log a run.
#
# Binding affinity in one command: pretraining, features and the six test sets, from a YAML config
# (task1-affinity_regression/configs/):
#   docker run --rm --gpus all --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD/dataset/data:/workspace/VoxBind/dataset/data \
#     -v $PWD/pretrain/exps:/workspace/VoxBind/pretrain/exps \
#     -v $PWD/pretrain/log:/workspace/VoxBind/pretrain/log \
#     -v $PWD/results:/workspace/VoxBind/results \
#     voxbind:env /opt/conda/envs/voxbind/bin/python task1-affinity_regression/run.py \
#     task1-affinity_regression/configs/ab7v2p1_clean.yaml
#
# Generated-ligand evaluation (Table 3 protocol) of a folder of target_NN/ directories:
#   docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD/dataset/data:/workspace/VoxBind/dataset/data \
#     -v $PWD/results:/workspace/VoxBind/results \
#     voxbind:env bash -c 'SAMPLE_DIR=results/task2-drugdesign/AR/samples bash script/05_evaluate.sh'

ARG MICROMAMBA_VERSION=1.5.10
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# git → AutoDockTools_py3 VCS pip pin; X libs → rdkit/openbabel.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates libxrender1 libxext6 libsm6 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /workspace/VoxBind

# conda-lock as the installer (uses micromamba as the solver).
RUN micromamba install -y -n base -c conda-forge conda-lock \
 && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy just the locks first so these heavy layers cache across code changes.
# voxbind: `conda-lock install` restores conda AND pip deps from the YAML lock.
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.conda-lock.yml /tmp/voxbind.conda-lock.yml
RUN micromamba run -n base conda-lock install --micromamba -n voxbind /tmp/voxbind.conda-lock.yml \
 && micromamba clean --all --yes

# voxeval: explicit conda lock, pinned pip layer, Vina 1.2.2 built from its sdist (the same function a
# native install runs).
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxeval.conda-linux-64.lock env/voxeval.pip.lock.txt /workspace/VoxBind/env/
COPY --chown=$MAMBA_USER:$MAMBA_USER script/00_setup_env.sh script/_common.sh /workspace/VoxBind/script/
RUN bash script/00_setup_env.sh voxeval \
 && micromamba clean --all --yes

# Project source (data/weights excluded via .dockerignore) + editable install into voxbind.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/VoxBind
RUN micromamba run -n voxbind pip install --no-cache-dir -e .

ENV ENV_NAME=voxbind \
    MAMBA_DEFAULT_ENV=voxbind \
    HOME=/home/$MAMBA_USER \
    VOXBIND_PY=/opt/conda/envs/voxbind/bin \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

CMD ["bash"]
