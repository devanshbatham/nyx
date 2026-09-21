# nyx

A compact 4B decision model, production gateway, and typed SDKs for Choice, Score, and Noul. The 3.29 GB GGUF runs through the pinned llama.cpp build on CPU, Apple Metal, NVIDIA CUDA, and AMD HIP. The public API is `POST /v1/systemone`.

![Animated light-theme chart comparing nyx and Jev across six classification benchmarks](assets/nyx-benchmarks.svg)

## Results

The table is the pre-quantization comparison. The pretrained Qwen base and nyx reference used the same local runtime, and all three systems received the same 2,277 frozen requests.

| Benchmark | Qwen3.5-4B base | nyx | Jev 1.13 |
|---|---:|---:|---:|
| AG News | **87.75%** | 85.75% | 86.50% |
| TREC coarse | 86.60% | 84.20% | **93.00%** |
| CoLA | 77.75% | **81.50%** | 77.75% |
| RTE | 84.12% | 88.09% | **91.34%** |
| Emotion | **47.67%** | 46.33% | 47.00% |
| IMDb | 95.75% | 95.75% | **97.00%** |
| Dataset-macro accuracy | 79.94% | 80.27% | **82.10%** |
| Pooled accuracy | 81.42% | 81.51% | **83.62%** |
| Round trip, median / p95 | 54.85 / 67.96 ms | 55.56 / 72.77 ms | 182.33 / 245.12 ms |
| Throughput | 17.44 req/s | 17.15 req/s | 42.43 req/s |

Local timings used one AMD Instinct MI325X, one execution slot, Q8 KV cache, Flash Attention, and a 4,096-token context. Jev used remote HTTPS at concurrency 8, so its timing is descriptive, not a hardware-controlled speed comparison. Both local runs and Jev completed every case without errors. See the [benchmark record](benchmarks/random-six.md).

The released GGUF is 3.29 GB, an importance-matrix IQ4_XS quantization with selected Q5_K/Q6_K tensors. On a separate 1,024-case fidelity set it agreed with the 8.42 GB reference on 98.73% of decisions (1,011/1,024) and scored 874/1,024 versus 873/1,024 for the reference. It missed the original 99.5% agreement target, so the public table above remains explicitly pre-quantization evidence rather than a direct score for the released file.

## Requirements

- 8 GB free RAM or unified memory for a single 4,096-token slot; more for larger contexts or concurrency
- 6 GB free disk for the model, source, and build
- A recent C++ compiler and CMake
- Optional: Apple Metal, NVIDIA CUDA, or AMD ROCm/HIP for acceleration

The model fits the 20 GB NVIDIA RTX 4000 Ada Droplet. DigitalOcean lists it at $0.76/GPU/hour in Toronto: [pricing](https://www.digitalocean.com/pricing/gpu-droplets), [availability](https://docs.digitalocean.com/products/droplets/details/gpu-availability/). A strict $500 ceiling permits about 657 billed hours, so it must be destroyed when idle to stay below that cap. NVIDIA latency and class-decision parity still require an on-device release check.

## Run

```bash
git clone https://github.com/devanshbatham/nyx.git /opt/nyx
hf download devanshbatham/nyx --local-dir /opt/nyx-model

python3 -m venv /opt/nyx/.venv
/opt/nyx/.venv/bin/pip install '/opt/nyx[server]'

# Choose cuda or hip.
NYX_ACCELERATOR=cuda /opt/nyx/scripts/build-llama-cpp.sh /opt/llama.cpp

export LLAMA_SERVER_BIN=/opt/llama.cpp/build/bin/llama-server
export NYX_GGUF_PATH=/opt/nyx-model/nyx.gguf
/opt/nyx/scripts/serve-model.sh
```

The model backend binds to unauthenticated loopback port `30000`; never expose it publicly. Start the authenticated gateway:

```bash
install -d -m 0750 /etc/nyx
/opt/nyx/.venv/bin/nyx-keygen --output /etc/nyx/api-key
set -a; . /opt/nyx/.env.example; set +a
/opt/nyx/.venv/bin/nyx-serve
```

Put TLS in front of port `8000`. Hardened systemd units are in [`deploy/`](deploy/).

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $(cat /etc/nyx/api-key)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nyx","state":"I loved it","questions":{"sentiment":{"type":"choice","instructions":"Classify sentiment","criteria":{"positive":null,"negative":null}}}}'
```

## TypeSafe drop-in

The gateway implements the TypeSafe System One wire contract for Choice, Score, Noul, model discovery, bearer authentication, and request IDs. CI tests the official Python SDK `0.7.0` end to end and contract-tests JavaScript SDK `0.6.0`. Applications change only their base URL and API key; responses identify the model as `nyx`. Wire compatibility does not imply behavioral identity with Jev.

```bash
export TYPESAFE_BASE_URL=https://nyx.example.com
export TYPESAFE_API_KEY="$(cat /etc/nyx/api-key)"
```

Native clients are included:

```bash
python -m pip install "git+https://github.com/devanshbatham/nyx.git"
npm install github:devanshbatham/nyx#main
```

## Verify

```bash
python -m pip install -e '.[server,test]'
pytest -q
npm ci && npm test
```

## License

Apache-2.0. nyx is derived from [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B); Qwen and Alibaba Cloud are credited as the original model authors. This project is not affiliated with Qwen, Alibaba Cloud, TypeSafe, or Jev.
