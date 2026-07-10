# Hybrid Token-Efficient Router

The router is optimized for token-scored evaluation: deterministic tools and a
locally evaluated answer are always attempted first. Fireworks is used only
when the local answer is unavailable or fails the acceptance policy; the router
then selects the cheapest configured model that supports the task, with
task-specific preference used as a tie-breaker.

## Pre-router flow

`sanity checks -> task detection -> difficulty estimation -> prompt library -> runtime budget -> tools -> local answer + evaluation -> cheapest Fireworks fallback`

## Local setup (Windows)

1. Install the Intel NPU driver and use Windows 11 22H2 or newer.
2. Create and activate an environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Export an OpenVINO INT4 model. A small instruct model is appropriate for
the pre-router workload:

```powershell
optimum-cli export openvino --model "Qwen/Qwen2.5-1.5B-Instruct" --weight-format int4 models\qwen2.5-1.5b-instruct-int4-ov
```

4. Run with the default NPU-first configuration, or override it:

```powershell
$env:LOCAL_MODEL_PATH = "models\qwen2.5-1.5b-instruct-int4-ov"
$env:LOCAL_PREFERRED_DEVICE = "NPU"
$env:LOCAL_FALLBACK_DEVICE = "CPU"
$env:FIREWORKS_API_KEY = "your-fireworks-api-key"
$env:ALLOWED_MODELS = "accounts/fireworks/models/model-a,accounts/fireworks/models/model-b"
$env:FIREWORKS_MODEL_CATALOG = '[{"id":"accounts/fireworks/models/model-a","input_cost_per_million":0.2,"output_cost_per_million":0.2,"max_difficulty":"medium","task_types":["ner","sentiment","summarization"]},{"id":"accounts/fireworks/models/model-b","input_cost_per_million":1.0,"output_cost_per_million":1.0,"max_difficulty":"hard","task_types":[]}]'
```

`FIREWORKS_MODEL_CATALOG` must use the actual allowed model IDs and current
token prices for the hackathon. The selector filters it by `ALLOWED_MODELS`,
excludes models that cannot satisfy the task difficulty, and minimizes estimated
input plus maximum output cost first. If two candidates cost the same, the
router prefers the model that is ranked best for the detected task type. Set
`LOCAL_ACCEPT_DIFFICULTIES=easy` (the default) to be conservative, or include
`medium` after validating local quality.

You can change the remote selection policy with `MODEL_SELECTION_MODE`:

```powershell
$env:MODEL_SELECTION_MODE = "cost_first"
$env:MODEL_SELECTION_MODE = "balanced"
$env:MODEL_SELECTION_MODE = "quality_first"
```

`cost_first` is the default. `balanced` keeps cost as the primary signal but
gives a little more weight to task fit. `quality_first` picks the most suitable
task model first and uses cost only as the fallback ordering.
Set `LOCAL_MODEL_ENABLED=false` where no exported model exists; the router then
falls back to Fireworks without failing.

OpenVINO keeps compiled artifacts in `.openvino_cache` to reduce later startup
time. If the NPU driver/device/model is incompatible, `LocalModel` records
`state.local_fallback_used = True` and serves the request from CPU.

The NPU plugin requires an Intel Core Ultra NPU and a compatible driver; the
application works on CPU without either. The standardized environment should
not depend on NPU availability because local-model errors route to Fireworks.
