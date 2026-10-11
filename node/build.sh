#!/usr/bin/env bash
# Build the node image locally for linux/amd64 and load it into the local docker. Never pushes.
# usage: node/build.sh [version]   (default: node/VERSION)
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${1:-$(cat VERSION)}"
REPO="${REPO:-ghcr.io/nmatare/pearl-node}"
for arch in ampere-ada blackwell; do
  docker buildx build --platform linux/amd64 --load --build-arg "PM_ARCH=${arch}" -t "${REPO}:${VERSION}-${arch}" -f Dockerfile .
done
echo
echo "Built ${REPO}:${VERSION}-ampere-ada and ${REPO}:${VERSION}-blackwell (local only, nothing pushed)."
echo "To publish (after: echo \$GHCR_TOKEN | docker login ghcr.io -u <user> --password-stdin):"
for arch in ampere-ada blackwell; do echo "  docker push ${REPO}:${VERSION}-${arch}"; done
