#!/usr/bin/env bash
set -euo pipefail

version="1.30.0"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
destination="${repo_root}/.deps/onnxruntime"

if [[ -e "${destination}" ]]; then
    echo "${destination} already exists; remove it yourself before reinstalling." >&2
    exit 1
fi

os="$(uname -s)"
arch="$(uname -m)"
case "${os}/${arch}" in
    Darwin/arm64)
        package="onnxruntime-osx-arm64-${version}"
        ;;
    Darwin/x86_64)
        package="onnxruntime-osx-x86_64-${version}"
        ;;
    Linux/x86_64)
        package="onnxruntime-linux-x64-${version}"
        ;;
    Linux/aarch64|Linux/arm64)
        package="onnxruntime-linux-aarch64-${version}"
        ;;
    *)
        echo "Unsupported platform: ${os}/${arch}" >&2
        exit 1
        ;;
esac

url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${package}.tgz"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "${temporary_dir}"' EXIT
mkdir -p "${repo_root}/.deps"

echo "Downloading ONNX Runtime ${version} for ${os}/${arch}..."
curl --fail --location --retry 3 "${url}" -o "${temporary_dir}/${package}.tgz"
tar -xzf "${temporary_dir}/${package}.tgz" -C "${temporary_dir}"
mv "${temporary_dir}/${package}" "${destination}"
echo "Installed in ${destination}"
