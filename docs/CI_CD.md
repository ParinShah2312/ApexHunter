# ApexHunter CI/CD

The pipeline is in [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).
It uses GitHub Actions because this project is hosted on GitHub. The deployable application is
the Streamlit entry point `frontend/app.py`, packaged with its Python backend in a Docker image.
No hosting account, server or live deployment is set up by these files.

## What happens automatically

```text
Push to any branch / open or update a PR / manual run
  -> build Python 3.12 CPU Docker image
  -> Ruff fatal-error checks
  -> pip dependency compatibility check
  -> existing unit tests
  -> start Streamlit and check its HTTP health endpoint
  -> on a push to main: publish that exact tested image to GHCR
  -> on a push to main: call a hosting webhook, if configured
```

Any failed check stops publication and deployment. Pull requests, other branches and manual runs
only perform checks. The existing runner discovers the unit files, including Streamlit AppTest
checks, and executes each file in its own process. `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` limit nested numerical-library threads on the CI runner.

Integration tests require the real, locally stored `data_lake/` and FastF1 `cache/`; this pipeline
does **not** run them or report them as passing. Run them locally with:

```bash
python backend/scripts/tests/run_tests.py --all
```

The HTTP smoke check proves that Streamlit starts and responds; it does not validate telemetry,
video playback, model output, or the presence of production data.

## Activate the pipeline

Commit these files and push them to GitHub. The `.github` directory must be at the Git repository
root (the inner `ApexHunter` folder in this checkout). Enable Actions if the repository has it
disabled. No added secret is needed for CI or image publication: GitHub creates `GITHUB_TOKEN`
for each run, and the workflow requests `packages: write` for the packaging job.

On a successful push to `main`, the images are:

```text
ghcr.io/parikhayush21/apexhunter:<full-commit-sha>
ghcr.io/parikhayush21/apexhunter:latest
```

The SHA tag identifies the source revision; `latest` moves to the most recently published image.
The owner/name are calculated from the repository, so forks publish under their own namespace.
If an existing GHCR package rejects publication, grant this repository Actions access in that
package's settings. Package visibility and read access for hosting are configured separately.

A local commit is not visible to GitHub until pushed. A push containing multiple commits runs
against the pushed branch tip. A newer push cancels an unfinished run for the same branch/event;
intermediate commits can be superseded. Production publication is intentionally limited to
`main`, so development branches cannot replace the production image.

## Explain the workflow YAML

YAML represents settings with `key: value`; indentation places settings under their parent.
Lines starting with `-` are list entries. `|` introduces a multiline shell script, while `>-`
folds multiple lines into one command. `${{ ... }}` is evaluated by GitHub Actions before a step
runs; `$LOCAL_IMAGE` is expanded later by Bash inside the runner.

| YAML entry | What it does here |
|---|---|
| `name` | Sets the label shown in GitHub's Actions tab. |
| `on.push.branches: ['**']` | Checks every branch push, including branch names containing `/`. |
| `on.pull_request` | Checks PRs when opened, reopened or updated. Uses `pull_request`, not the privileged `pull_request_target` event. |
| `on.workflow_dispatch` | Adds a manual **Run workflow** button; manual runs check without publishing. |
| `permissions` | Gives repository read access by default. Only the packaging job requests package write access; the deployment job has no token permissions. |
| `concurrency` | Groups runs by workflow, event and branch/PR; a newer run cancels the previous unfinished run in that group. |
| `jobs` | Defines the packaging job and optional deployment job. |
| `runs-on: ubuntu-latest` | Uses a fresh GitHub-hosted Linux runner with Docker. |
| `timeout-minutes` | Bounds the build/test and deployment jobs' running time. |
| `env` | Defines values available to commands, including the local image name. |
| `steps` | Lists ordered actions and commands; a failing step stops later normal steps. |
| `uses` | Runs an existing action: checkout, Buildx, image build or registry login. |
| `with` | Supplies inputs to that action. Checkout disables persisted Git credentials. |
| `load: true`, `push: false` | Loads the built image into the runner's Docker engine without publishing yet. |
| `cache-from`, `cache-to` | Reuses Docker build layers through GitHub's cache to speed up later builds. |
| `run` | Runs Bash commands: Ruff, tests, health check, image tags/pushes or webhook call. |
| `if` | Allows registry login/publication and deployment only for pushes to `main`. |
| `github.repository`, `github.sha` | Supply the repository name and exact source commit to image naming and metadata. |
| `secrets.GITHUB_TOKEN` | Provides GitHub's temporary token for GHCR authentication. No Docker Hub credentials are used. |
| `needs: verify-and-package` | Makes deployment wait for successful checks and image publication. |
| `secrets.DEPLOY_WEBHOOK_URL` | Optional hosting-provider URL. An absent value records an explicit deployment skip. |
| `GITHUB_STEP_SUMMARY` | Writes published image names or deployment status to the run's summary. |

The smoke-test script uses `trap` to remove its temporary container even if the check fails.
Publication retags and pushes the already tested image; it does not rebuild after testing.

## Connect deployment later (optional)

Choose a hosting service that accepts an HTTPS POST deployment hook, then configure a service
to pull this GHCR image's `latest` tag. Provide registry read credentials if the package is
private, and configure the service's port as **8501**. Disable independent Git-triggered deploys
if you want the CI checks to gate every deployment. Store the hook URL in GitHub under
**Settings -> Secrets and variables -> Actions -> New repository secret** as:

```text
DEPLOY_WEBHOOK_URL
```

After that, each successful push to `main` publishes the image and requests a hosting deployment.
The hook must deploy the configured GHCR image; the workflow sends an empty POST, not an image
name or credentials. Providers requiring a custom payload/API need a provider-specific deploy
step. A successful HTTP response means the request was accepted; the hosting provider must
report rollout completion and application health. With no hook, this is CI plus continuous
delivery of an image, with live deployment skipped.

## Docker and persistent data

The Dockerfile uses Python 3.12, CPU-only PyTorch/torchvision, native OpenCV libraries, the
existing `requirements.txt`, a non-root user (UID 10001), and Streamlit on `0.0.0.0:8501`.
It preserves the existing Streamlit configuration, including static serving and video message
size. The health check uses `/_stcore/health`.

`.dockerignore` excludes secrets, Git history, caches, models and the large data lake. Backend,
frontend, requirements and Streamlit configuration are explicitly copied into the image.
The tracked `models/best.pt` is also excluded: supply required models through persistent storage.
Mount your existing data at `/app/data_lake`, models at `/app/models`, and a writable FastF1 cache
at `/app/cache`. On Linux, grant the container's UID 10001 access; ensure the cache is writable.
Without cleaned telemetry, the dashboard displays its missing-data message even when the
Streamlit health endpoint is healthy. The pipeline never downloads the full dataset or retrains
models. Hadoop, MongoDB and Spark services are not provisioned by this container.

To try the same container locally with Docker running, from the repository root:

```bash
docker build -t apexhunter:local .
docker run --rm -p 8501:8501 apexhunter:local
```

For a Linux server with existing data, mount the three folders (replace `/srv/apexhunter`):

```bash
docker run --rm -p 8501:8501 \
  --mount type=bind,src=/srv/apexhunter/data_lake,dst=/app/data_lake,readonly \
  --mount type=bind,src=/srv/apexhunter/models,dst=/app/models,readonly \
  --mount type=bind,src=/srv/apexhunter/cache,dst=/app/cache \
  apexhunter:local
```

The dependencies in the existing `requirements.txt` remain unpinned, and the base image and
actions use maintained version tags. A clean future build can resolve different dependency
versions; dependency compatibility, tests and startup are checked on every build. The SHA image
tag preserves the produced artifact for that revision. Dependabot proposes weekly action and
base-image updates; Python dependency locking is separate future work.

## Validation of this change

Local checks cover workflow YAML structure, Bash syntax, Dockerfile/configuration review,
Ruff's fatal-error checks, and whitespace checks. Docker Engine was not running locally and the
local Python environment did not have the application dependencies installed, so full execution
was verified on GitHub's Linux runner.

First GitHub run (30 Oct 2026): Docker build, Ruff and dependency checks passed. All 127 unit
tests ran with zero skips; one existing AppTest selector failed because it assumed HTML blocks
were `UnknownElement` instances. The test now uses the public `at.get("html")` API without
weakening its assertions. The [follow-up run](https://github.com/parikhayush21/ApexHunter/actions/runs/41034991153)
passed Docker build, Ruff, dependency checks, all 127 unit tests in 23 files (zero skips), and the
Streamlit HTTP startup check. GHCR publication runs only on a push to `main`; hosting remains
unconfigured and no live deployment has been performed.

References: [GitHub workflow syntax](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions),
[GitHub container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry),
[Streamlit Docker deployment](https://docs.streamlit.io/deploy/tutorials/docker).
