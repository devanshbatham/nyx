# Nyx SDK

Production clients and an authenticated decision gateway for
[`devanshbatham/nyx`](https://huggingface.co/devanshbatham/nyx). The model is a
task-specialized Qwen3.5-35B-A3B checkpoint in its qualified INT4+FP8 SGLang format.

The repository contains no weights and no playground.

## Architecture

```text
application -> TLS proxy -> Nyx gateway :8000 -> private SGLang :30000
```

Keep both processes on loopback or a private network. The gateway provides bearer authentication,
request/body bounds, rate and concurrency limits, deadlines, cancellation, and typed Choice, Score,
and Noul responses. TLS termination is intentionally left to the production ingress.

## Python client

```bash
python -m pip install "git+https://github.com/devanshbatham/nyx.git"
export NYX_API_KEY="$(cat /etc/nyx/api-key)"
export NYX_BASE_URL="https://nyx.example.com"
```

```python
from nyx import NyxClient

with NyxClient() as client:
    result = client.systemone(
        state={"message": "I was charged twice."},
        questions={
            "route": {
                "type": "choice",
                "instructions": "Route this support request.",
                "criteria": {
                    "billing": "Payments, invoices, or charges",
                    "other": "Anything else",
                },
            }
        },
    )
```

`AsyncNyxClient` has the same methods for asynchronous applications. Do not put bearer keys in
browser or other untrusted client code.

## TypeScript client

The dependency-free Node/browser client is in [`typescript/`](typescript/). Install it from this
repository until a registry release is published:

```bash
npm install github:devanshbatham/nyx#main
```

```ts
import { NyxClient } from "@devanshbatham/nyx";

const nyx = new NyxClient({
  apiKey: process.env.NYX_API_KEY!,
  baseUrl: process.env.NYX_BASE_URL!,
});

const result = await nyx.systemone({
  state: { message: "I was charged twice." },
  questions: {
    route: {
      type: "choice",
      instructions: "Route this support request.",
      criteria: { billing: "Payments or charges", other: "Anything else" },
    },
  },
});
```

## Gateway deployment

Requirements: Linux, Python 3.11+, and the Nyx SGLang backend already listening privately. Follow
the model repository's pinned runtime instructions first.

```bash
git clone https://github.com/devanshbatham/nyx.git /opt/nyx-sdk
python3 -m venv /opt/nyx-sdk/.venv
/opt/nyx-sdk/.venv/bin/pip install '/opt/nyx-sdk[server]'
install -d -m 0750 -o nyx -g nyx /etc/nyx
sudo -u nyx /opt/nyx-sdk/.venv/bin/nyx-keygen --output /etc/nyx/api-key
```

Download the model repository to `/opt/nyx-model`, copy `.env.example` to
`/etc/nyx/gateway.env`, and adjust paths and limits. Test the foreground service:

```bash
set -a
. /etc/nyx/gateway.env
set +a
/opt/nyx-sdk/.venv/bin/nyx-serve
```

For systemd, install [`deploy/nyx-gateway.service`](deploy/nyx-gateway.service), create the
unprivileged `nyx` service account, then enable the unit. Put an HTTPS reverse proxy in front of
`127.0.0.1:8000`; never expose port `30000`.

Health and inference:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $(cat /etc/nyx/api-key)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nyx","state":"I loved it","questions":{"sentiment":{"type":"choice","instructions":"Classify sentiment","criteria":{"positive":null,"negative":null}}}}'
```

The unauthenticated health route verifies the SGLang backend. All other routes require the bearer
key. `/internal/metrics` returns process-local JSON metrics and must remain behind trusted ingress.

## Configuration

The supported production settings are documented in [`.env.example`](.env.example). The gateway
fails closed when no key hash or key file is supplied, and it rejects a backend whose reported
model path does not match `NYX_BACKEND_MODEL_PATH`.

## Verification

```bash
python -m pip install -e '.[server,test]'
pytest
npm ci && npm test
```

## License

Apache-2.0. Nyx is derived from
[`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B); Qwen and Alibaba Cloud are
credited as the original model authors. Model weights are distributed separately under their model
repository terms.
