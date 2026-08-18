FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        clang-format \
        cmake \
        curl \
        gdb \
        git \
        ninja-build \
        pciutils \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Fail the image build immediately if the CUDA development toolkit is missing.
RUN nvcc --version

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=1001

RUN mkdir -p /workspaces && \
    groupadd --gid ${USER_GID} ${USERNAME} || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} || true && \
    chmod 755 /home/${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV CODEX_HOME="/workspaces/.codex-logs"
RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh

WORKDIR /workspaces
