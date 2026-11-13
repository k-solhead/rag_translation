# ---------------------------------------------------------------
# RAG Translation App（Streamlit）
# ---------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

RUN useradd --create-home appuser

# CPU 版 PyTorch を先に入れる（指定しないと GPU 版が入り、イメージが数GB増える）
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# 埋め込みモデルをイメージに同梱（起動時のダウンロード不要・オフラインで動作）
ARG TM_EMBED_MODEL=intfloat/multilingual-e5-small
ENV TM_EMBED_MODEL=${TM_EMBED_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['TM_EMBED_MODEL'])" \
 && chown -R appuser /opt/hf
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY *.py ./

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
