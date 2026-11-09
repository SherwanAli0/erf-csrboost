# Runs on Fly.io (see DEPLOY_FLY.md). The app binds to $PORT (7860).
FROM python:3.11-slim AS package-build
WORKDIR /build
COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt
COPY evaluation/ ./evaluation/
RUN python -m build --no-isolation --wheel --outdir /wheels ./evaluation

FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PORT=7860 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
WORKDIR /app
ARG FYD_RELEASE_ID=development
ENV FYD_RELEASE_ID=$FYD_RELEASE_ID

COPY --chown=user requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY --from=package-build --chown=user /wheels/ /wheels/
RUN pip install --no-cache-dir --no-deps /wheels/*.whl

COPY --chown=user backend/ ./backend/
COPY --chown=user frontend/ ./frontend/
COPY --chown=user samples/ ./samples/
COPY --chown=user LICENSE LICENSE-CC-BY-4.0.txt THIRD_PARTY_NOTICES.md ./
# The wheel owns the single copy of the workbench assets in production.
ENV FYD_WEB_DIR=/home/user/.local/lib/python3.11/site-packages/rare_event_eval/web

EXPOSE 7860
CMD ["python", "backend/api.py"]
