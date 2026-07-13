# Deploying AvoGrade

The service is a FastAPI app that loads the trained ResNet once at startup and
grades uploaded avocado photos. These steps take it from localhost to a public
URL on real cloud infrastructure.

## 0. Files to add to the repo

Copy these into your project root:

- `Dockerfile`          -> project root
- `.dockerignore`       -> project root
- `.github/workflows/ci.yml` -> creates `.github/workflows/` and the CI file

Also add `python-multipart` to the `serve` extra in `pyproject.toml` so uploads
work outside Docker too:

    serve = ["fastapi>=0.110", "uvicorn>=0.27", "pillow>=10.0", "python-multipart>=0.0.9"]

## 1. Build the image locally

    docker build -t avograde .

First build is slow (downloads torch); expect a ~1.5-2 GB image — normal for a
PyTorch service. The model weights are baked in, so the container is self-contained.

## 2. Test the container locally (do this BEFORE deploying)

    docker run -p 8000:8000 avograde

Then in another terminal:

    curl http://localhost:8000/healthz
    curl -X POST http://localhost:8000/grade -F "image=@some_avocado.jpg"

If you get a grade back from inside the container, it's ready to ship. This
local check is the step that catches problems before they cost you a cloud debug.

## 3a. Deploy to Google Cloud Run (recommended — real, serverless infra)

Cloud Run runs your container, scales to zero when idle (cheap), and gives a
public HTTPS URL. It builds from your Dockerfile via Cloud Build.

    gcloud run deploy avograde \
      --source . \
      --region europe-west4 \
      --allow-unauthenticated \
      --memory 2Gi \
      --cpu 1 \
      --port 8000

Notes:
- `--memory 2Gi` matters: torch + ResNet won't fit in the 512Mi default.
- The `.pt` weights upload with the source because they're in the build context
  (and not in `.dockerignore`).
- Cold starts take a few seconds while the model loads — expected for scale-to-zero.
- The command prints a `https://avograde-....run.app` URL when done. That's your
  live demo link.

## 3b. Alternative: Hugging Face Spaces (easiest, ML-friendly, free)

Create a Space with the "Docker" SDK, push this repo (plus the `.pt`), and it
builds and hosts the same Dockerfile automatically. Good if you don't want a
cloud account.

## What to say about it (the senior story)

- Containerized and reproducible; CPU-only image to stay lean.
- Tests run in CI on every push.
- Model loaded once at startup; `/healthz` for liveness; latency measured per request.
- Scales to zero on Cloud Run; cold-start cost is the model load (~seconds), which
  is why the cache and fallback matter under load.
