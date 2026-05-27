FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY fintech_dashboard ./fintech_dashboard
COPY config ./config
COPY data ./data
COPY models ./models
COPY src ./src
COPY THESIS_CHAPTERS.md ./THESIS_CHAPTERS.md
COPY README.md ./README.md

EXPOSE 8501

RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["streamlit", "run", "fintech_dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
