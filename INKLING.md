# Inkling 2 Mini on oMLX with fused decode kernels

This branch (`inkling2-fast-decode`, based on oMLX 0.8.1) adds fused Metal decode kernels for Inkling 2 Mini
and a script that re-quantizes its attention weights to 6 bits. Measured on an M5 Max (40-core GPU,
128 GB) on AC power, temperature 0, 500-token replies to short prompts:

| Setup | Decode |
|---|---:|
| oMLX 0.8.1, oQ4e model | 45–50 tok/s |
| Fused decode kernels + 6-bit attention, MTP off | 62–64 tok/s |
| Fused decode kernels + 6-bit attention, MTP on | 55–62 tok/s |

Speeds vary by a few tok/s between runs. With the fused kernels, MTP drafting costs about as much as it
saves, because verifying draft tokens reads more experts. The output matches the unfused path to within
rounding (KL divergence about 0.001).

## Memory

The model takes about 17.5 GB once loaded, plus the KV cache for long prompts. A Mac with 32 GB of unified
memory runs it; 64 GB leaves plenty of room for other applications.

## 1. Requirements

- Apple silicon Mac, macOS 15 or later.
- Xcode Command Line Tools (`xcode-select --install`) and [uv](https://docs.astral.sh/uv/).
- Hugging Face CLI (`uv tool install huggingface_hub`), about 40 GB of free disk.

## 2. Install oMLX from this branch

```sh
git clone -b inkling2-fast-decode https://github.com/jarrelscy/omlx.git ~/src/omlx
cd ~/src/omlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
# The native kernels need Xcode's Metal compiler to build; this copies the prebuilt ones from the 0.8.1 wheel.
./scripts/inkling/install_native_kernels.sh
```

Check that all five kernel sets load:

```sh
.venv/bin/python -c "from omlx.custom_kernels import native_kernel_status as s; print({k: v.get('available') for k, v in s().items()})"
```

## 3. Download the model and re-quantize attention to 6 bits

```sh
hf download mlx-community/Inkling-2-Mini-oQ4e-mtp --local-dir ~/Models/Inkling/Inkling-2-Mini-oQ4e-mtp
.venv/bin/python scripts/inkling/requant_attention.py \
    ~/Models/Inkling/Inkling-2-Mini-oQ4e-mtp ~/Models/Inkling/Inkling-2-Mini-oQ4e-a6-mtp
```

The script re-quantizes the 56 attention projections from 8 to 6 bits and links the other files from the
download. It takes a few minutes.

## 4. Register and start

```sh
mkdir -p ~/.omlx/models
ln -s ~/Models/Inkling/Inkling-2-Mini-oQ4e-a6-mtp ~/.omlx/models/inkling-2-mini
.venv/bin/omlx serve --model-dir ~/.omlx/models --host 127.0.0.1 --port 8000
```

The server answers on `http://127.0.0.1:8000/v1` (OpenAI chat completions). To turn MTP drafting on or off,
set `mtp_enabled` for `inkling-2-mini` in `~/.omlx/model_settings.json` (or in the oMLX admin page) and
restart.

`OMLX_INKLING2_FAST_DECODE=0` turns the fused kernels off, for comparison.

## 5. Requests

Inkling's chat template ignores `enable_thinking`. It reads `reasoning_effort` (0.0 to 0.99, default 0.9):

```json
{"model": "inkling-2-mini", "messages": [...], "chat_template_kwargs": {"reasoning_effort": 0}}
```

At 0 it still writes a short reasoning block. FrontierOS (branch `local-inkling`) sends this setting and uses
native tool calls with Inkling.
