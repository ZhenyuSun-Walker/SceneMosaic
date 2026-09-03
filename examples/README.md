# Public examples

images/ contains four image inputs used to exercise SceneMosaic's complete
public pipeline. They are inputs only: no model checkpoint, API credential,
cache, intermediate artifact, or generated scene is included here.

| Name | Image | Command selector |
| --- | --- | --- |
| Scene 018 | images/scene_018.png | scene_018 |
| Scene 031 | images/scene_031.png | scene_031 |
| Scene 092 | images/scene_092.png | scene_092 |
| Gaming lounge | images/gaminglounge.png | gaminglounge |

After completing the repository setup, create the workspace and machine-local
runtime profile. The configurator resolves the default named Conda
environments, so no interpreter path needs to be edited:

~~~bash
mkdir -p workspace/perception_attempts
python scripts/configure_runtime.py
export OPENAI_API_KEY="replace-with-your-key"
# Optional for an OpenAI-compatible gateway:
# export OPENAI_BASE_URL="https://gateway.example/v1"
conda run -n scenemosaic python -m vlm_agent.cli verify
~~~

When OPENAI_BASE_URL is unset, perception, relation extraction, and evolution
all use the official OpenAI gateway.

Run a new complete scene with a fresh name:

~~~bash
conda run --no-capture-output -n scenemosaic \
  python examples/run_end_to_end.py \
  --example scene_092 \
  --scene-name scene_092_demo
~~~

run_end_to_end.py delegates directly to python -m vlm_agent.cli run; it does
not reimplement any pipeline stage. The run performs image inventory, SAM3
segmentation, repository-local SAM 3D Objects reconstruction, relation
extraction, and the native scene-evolution stage. It uses the configured GPU
and the VLM services named in the local profile.

The configured workspace receives runs/<scene-name>/. Its reconstruction/
directory contains the perception and SAM3D artifacts, relations/ contains the
accepted relation publication, and the scene root contains the final evolution
publication plus outputs/representative_scenes/.

`configs/runtime.local.json` is intentionally left untracked. The
configurator creates it from `configs/runtime.example.json`, resolves the
installed Conda interpreters, and leaves repository asset paths relative.
Keeping it untracked allows machine-local environment locations without
publishing them. The status command below assumes that setup step has already
been completed.

To inspect a completed or resumable run, use:

~~~bash
conda run -n scenemosaic python -m vlm_agent.cli \
  --runtime-config configs/runtime.local.json \
  status --scene-root workspace/runs/scene_092_demo
~~~

For a custom image, replace --example scene_092 with
--input-image path/to/image.png. The command intentionally requires an
explicit scene name and never stores credentials in command-line arguments.
