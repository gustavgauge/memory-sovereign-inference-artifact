#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {qwen3-next|gemma4} DESTINATION" >&2
  exit 2
fi

identity=$1
destination=$2
case "$identity" in
  qwen3-next)
    base=8ef78e644f559db4e8716b59bf76b8e11619337d
    patch=runtime/qwen3-next/patches/0001-Add-Qwen3-Next-streamed-expert-destination-hook.patch
    ;;
  gemma4)
    base=40d5358d3c730b81729ba81cd5c44ed596d02510
    patch=runtime/gemma4/patches/0001-Add-Gemma-streamed-expert-destination-hook.patch
    ;;
  *)
    echo "unknown identity: $identity" >&2
    exit 2
    ;;
esac

if [[ -e "$destination" ]]; then
  echo "destination already exists: $destination" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
git clone https://github.com/ggml-org/llama.cpp.git "$destination"
git -C "$destination" checkout --detach "$base"
git -C "$destination" am "$root/$patch"
echo "Reconstructed $identity source at $(git -C "$destination" rev-parse HEAD)"
