# syntax=docker/dockerfile:1.6
# Generic node image: node/agent.py runs the plan's workloads (Pearl miner now; Dolphin worker is supplied at runtime).
# CUDA base only provides env/ld paths; the NVIDIA driver (libcuda) is injected by the host. Never overwrite LD_LIBRARY_PATH.
# One build, tagged per arch (ampere-ada, blackwell) so groups can diverge later (research/miner.md section 6).
# CUDA base per arch: Salad only schedules on hosts whose driver satisfies the image's NVIDIA_REQUIRE_CUDA, so Ampere/Ada
# uses 12.2 (most gaming-PC drivers) and only Blackwell needs 12.8. CI passes CUDA_TAG and CUDA_PKG per variant.
ARG CUDA_TAG=12.2.2
FROM nvidia/cuda:${CUDA_TAG}-base-ubuntu22.04
ARG CUDA_PKG=12-2

ARG SRB_VER=3.7.5
ARG SRB_SHA=f18e33dac31d4fd5c6909409d7a82793477e8b93c63b897c7a19befff0c33ca8
ARG PM_ARCH=ampere-ada

# The tarball name uses dashes (3-7-5), the tag dots. ADD --checksum fails the build on a sha256 mismatch.
ADD --checksum=sha256:${SRB_SHA} \
    https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRB_VER}/SRBMiner-Multi-3-7-5-Linux.tar.gz /tmp/srb.tgz

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates tini \
    # SRBMiner JIT-compiles its pearlhash kernels: without NVRTC it connects to the pool but never hashes (Vast A100 smoke)
    cuda-nvrtc-${CUDA_PKG} cuda-cudart-${CUDA_PKG} \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/srb /opt/dolphin \
 && tar -xzf /tmp/srb.tgz -C /opt/srb --strip-components=1 \
 && ln -s /opt/srb/SRBMiner-MULTI /usr/local/bin/SRBMiner-MULTI \
 && rm /tmp/srb.tgz

# The Dolphin runtime bundle (Python 3.12 + conda-forge libs such as libicu 78) is built against GCC 14's libstdc++ and the
# worker starts its engine with LD_LIBRARY_PATH=...:/usr/lib/x86_64-linux-gnu, so the SYSTEM libstdc++ must provide
# CXXABI_1.3.15. Ubuntu 22.04 ships GCC 12's ("libstdc++.so.6: version `CXXABI_1.3.15' not found (required by libicui18n.so.78)",
# found on a live Vast A100: the engine died on `import sqlite3`, the worker only said "inference backend exited").
# Same glibc, newer C++ runtime from the toolchain PPA; the build fails if it is still too old.
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg gpg-agent dirmngr \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
 && apt-get purge -y --auto-remove software-properties-common \
 && rm -rf /var/lib/apt/lists/* \
 && grep -aq 'CXXABI_1.3.15' /usr/lib/x86_64-linux-gnu/libstdc++.so.6

COPY agent.py workername.py prefetch.py /opt/node/
COPY drivers/ /opt/node/drivers/

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PM_NODE_ARCH=${PM_ARCH}
LABEL org.opencontainers.image.source="https://github.com/nmatare/pearl_mining" pm.srbminer="${SRB_VER}" pm.arch="${PM_ARCH}"

# tini reaps orphans and forwards SIGTERM to the agent (which stops its children gracefully).
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "/opt/node/agent.py"]
