#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
model_dir="${repo_root}/models"
mkdir -p "${model_dir}"

download_checked() {
    local url="$1"
    local destination="$2"
    local expected_sha256="$3"
    local temporary_file="${destination}.download"

    if [[ -f "${destination}" ]]; then
        echo "Found ${destination}; checking checksum."
    else
        echo "Downloading $(basename "${destination}")..."
        curl --fail --location --retry 3 "${url}" -o "${temporary_file}"
        mv "${temporary_file}" "${destination}"
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status
    elif command -v shasum >/dev/null 2>&1; then
        printf '%s  %s\n' "${expected_sha256}" "${destination}" | shasum -a 256 --check --status
    else
        echo "Install sha256sum or shasum to verify model files." >&2
        exit 1
    fi
}

download_checked \
    "https://huggingface.co/onnxmodelzoo/resnet18-v1-7/resolve/e7cb849/resnet18-v1-7.onnx" \
    "${model_dir}/resnet18.onnx" \
    "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"

download_checked \
    "https://huggingface.co/onnxmodelzoo/mobilenetv2-7/resolve/b055c14/mobilenetv2-7.onnx" \
    "${model_dir}/mobilenet.onnx" \
    "c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f"

labels_file="${model_dir}/imagenet_labels.txt"
if [[ ! -f "${labels_file}" ]]; then
    curl --fail --location --retry 3 \
        "https://s3.amazonaws.com/onnx-model-zoo/synset.txt" \
        -o "${labels_file}.download"
    mv "${labels_file}.download" "${labels_file}"
fi

echo "Models and labels are ready in ${model_dir}."
