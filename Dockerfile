FROM python:3.12-slim

WORKDIR /home/purple

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (separate layer for caching; avoids pulling CUDA wheels).
RUN pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/tmp/purple_workspace
ENV OPENAI_MODEL=gpt-5-mini
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
