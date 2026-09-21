# Benchmarks

Six frozen classification tasks, one runner, and no stored credentials or provider responses.

```bash
python -m pip install -r benchmarks/requirements.txt
python benchmarks/run.py prepare

export NYX_API_KEY=...
python benchmarks/run.py run \
  --name nyx \
  --url http://127.0.0.1:8000/v1/systemone \
  --model nyx \
  --api-key-env NYX_API_KEY \
  --concurrency 1

export TYPESAFE_API_KEY=...
python benchmarks/run.py run \
  --name jev \
  --url https://api.typesafe.ai/v1/systemone \
  --model jev-1.13.0 \
  --api-key-env TYPESAFE_API_KEY \
  --concurrency 8

python benchmarks/run.py compare --left nyx --right jev
```

`tasks.json` pins every dataset revision, label, prompt, sampling rule, seed, case count, and expected case hash. Generated cases and results stay in `benchmarks/runs/`.
