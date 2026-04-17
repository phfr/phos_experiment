# JupyterLite on Coolify

This folder builds a **static JupyterLite site** (browser-only Python via Pyodide) and serves it with **Python’s standard library** (`serve.py`: `http.server` + correct `.wasm` MIME type). **You do not need nginx** unless you want it for gzip, custom headers, or team familiarity.

## Repository layout

The **Git repository root is the `v3/` tree** (same folder as `all.ipynb`, `data.csv`, `requirements.txt`). Docker `COPY` paths are written for that layout.

## Coolify settings

- **Build context:** repository root (`.`).
- **Dockerfile:** `jupyterlite/Dockerfile`
- **Port:** `8000` by default (`EXPOSE 8000`). If Coolify injects `PORT`, this image respects it.

## About the “simple Dockerfile” pattern

Using **`python -m http.server --directory ./_output`** is fine **if** `.wasm` is served as `application/wasm`. Some Python / OS MIME tables omit that; Pyodide can break in strict browsers. This repo uses **`serve.py`** so the WASM type is explicit.

**`RUN pip install -r requirements.txt` before `jupyter lite build` does *not* install packages into the browser kernel.** That only affects the **Linux** Python used during the image build. The notebook in the browser still uses **Pyodide** and its wheel index unless you add **piplite** wheels / Pyodide-compatible packages at build time. So a server `requirements.txt` is usually **not** the right lever for JupyterLite (and a heavy `requirements.txt` just bloats the builder layer).

## Optional assets in the image

The Dockerfile copies `all.ipynb` and `data.csv` into the Lite `content/` tree. For GMT enrichment, kinase maps, or other local files, add `COPY` lines before `jupyter lite build`.

## `pip install` at the top of the notebook?

**Not recommended** for production. Prefer build-time configuration and Pyodide-compatible dependencies.

## Pyodide vs server Jupyter

`all.ipynb` targets normal CPython (`requirements.txt`). Under JupyterLite, packages such as **hdbscan**, **gseapy**, or native extensions may be **missing or different**. For **full parity**, use **JupyterLab** in a container with `requirements.txt` instead of JupyterLite.

## Local smoke build

From the **repository root**:

```powershell
pip install -r jupyterlite/requirements-build.txt
New-Item -ItemType Directory -Force content | Out-Null
Copy-Item all.ipynb, data.csv content\
jupyter lite build --contents content --output-dir _output
$env:JUPYTERLITE_HTTP_ROOT = (Resolve-Path _output).Path
$env:PORT = "8000"
python jupyterlite\serve.py
```

In Docker, `JUPYTERLITE_HTTP_ROOT` defaults to `/srv` (where `_output` is copied).
