# CUDA 12.4 runtime + PyTorch 2.5.1; works with NVIDIA driver >= 550. Falls back to CPU without --gpus.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    MPLBACKEND=Agg \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY tools ./tools
COPY run_all.py config.yaml config.smoke.yaml ./

# outputs/, checkpoints/, cache/, dataset/ are mounted as volumes by run.sh / docker-compose
ENTRYPOINT ["python", "run_all.py"]
CMD ["--config", "config.yaml"]
