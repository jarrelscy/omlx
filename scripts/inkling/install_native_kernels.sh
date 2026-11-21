#!/bin/zsh
# Installs oMLX's prebuilt native kernels (Python extensions and Metal libraries) into a source checkout.
# A source build needs Xcode's Metal compiler to make them; with only the Command Line Tools they are
# missing and oMLX falls back to slower paths. This takes them from the 0.8.1 release wheel for Python 3.12,
# the release this branch is based on.
set -euo pipefail
cd "${0:A:h}/../.."
url=https://github.com/jundot/omlx/releases/download/v0.8.1/omlx-0.8.1-cp312-cp312-macosx_15_0_universal2.whl
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL -o "$work/omlx.whl" "$url"
unzip -oq "$work/omlx.whl" 'omlx/custom_kernels/*' -d "$work/whl"
count=0
for f in "$work"/whl/omlx/custom_kernels/**/(_ext*.so|*.metallib)(N); do
    rel=${f#$work/whl/}
    mkdir -p "${rel:h}"
    cp "$f" "$rel"
    count=$((count + 1))
done
echo "Installed $count native kernel files into omlx/custom_kernels."
