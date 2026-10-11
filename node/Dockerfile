# syntax=docker/dockerfile:1.6
# Generic node image: node/agent.py runs the plan's workloads (Pearl miner now; Dolphin worker is supplied at runtime).
# CUDA base only provides env/ld paths; the NVIDIA driver (libcuda) is injected by the host. Never overwrite LD_LIBRARY_PATH.
# One build, tagged per arch (ampere-ada, blackwell) so groups can diverge later (research/miner.md section 6).
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG SRB_VER=3.7.5
ARG SRB_SHA=f18e33dac31d4fd5c6909409d7a82793477e8b93c63b897c7a19befff0c33ca8
ARG PM_ARCH=ampere-ada

# The tarball name uses dashes (3-7-5), the tag dots. ADD --checksum fails the build on a sha256 mismatch.
ADD --checksum=sha256:${SRB_SHA} \
    https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRB_VER}/SRBMiner-Multi-3-7-5-Linux.tar.gz /tmp/srb.tgz

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/srb /opt/dolphin \
 && tar -xzf /tmp/srb.tgz -C /opt/srb --strip-components=1 \
 && ln -s /opt/srb/SRBMiner-MULTI /usr/local/bin/SRBMiner-MULTI \
 && rm /tmp/srb.tgz

COPY agent.py workername.py /opt/node/
COPY drivers/ /opt/node/drivers/

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PM_NODE_ARCH=${PM_ARCH}
LABEL org.opencontainers.image.source="https://github.com/nmatare/pearl_mining" pm.srbminer="${SRB_VER}" pm.arch="${PM_ARCH}"

# tini reaps orphans and forwards SIGTERM to the agent (which stops its children gracefully).
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "/opt/node/agent.py"]
