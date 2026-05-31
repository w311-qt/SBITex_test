FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-extras.txt ./
# CPU-only torch first so sentence-transformers does not pull the ~2.5GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt -r requirements-extras.txt

COPY rag_eval/ ./rag_eval/

ENTRYPOINT ["python", "-m", "rag_eval"]
CMD ["--help"]
