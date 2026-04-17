# JupyterLite static site for Coolify (or any container host).
# Build context: repository root (this project treats that root as `v3/`).
#
# Coolify: build context = repo root (`.`), Dockerfile `jupyterlite/Dockerfile`,
# expose container port 8000 (or set PORT to match your published port).

FROM python:3.12-slim AS builder

WORKDIR /build

COPY jupyterlite/requirements-build.txt /tmp/requirements-build.txt
RUN pip install --no-cache-dir -r /tmp/requirements-build.txt

RUN mkdir -p content
COPY all.ipynb content/all.ipynb
COPY data.csv content/data.csv

RUN jupyter lite build --contents content --output-dir _output

# Runtime: only Python stdlib + static files (no nginx required).
FROM python:3.12-slim

WORKDIR /srv
COPY --from=builder /build/_output ./
COPY jupyterlite/serve.py /srv/serve.py

ENV JUPYTERLITE_HTTP_ROOT=/srv
ENV PORT=8000
EXPOSE 8000

CMD ["python", "/srv/serve.py"]
