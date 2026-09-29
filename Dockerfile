FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
COPY --from=builder /wheels /wheels
COPY requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --constraint /tmp/requirements.lock /wheels/*.whl \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 researcher \
    && mkdir /results && chown researcher /results
WORKDIR /app
COPY configs/demo.toml /app/configs/demo.toml
USER researcher
ENTRYPOINT ["quant-research"]
CMD ["--config", "/app/configs/demo.toml", "--output", "/results/demo"]
