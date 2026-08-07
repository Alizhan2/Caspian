FROM python:3.11-slim
WORKDIR /workspace/backend
ARG INSTALL_ML=false
COPY requirements.txt requirements-ml.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 && \
      pip install --no-cache-dir -r requirements-ml.txt -c constraints.txt; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
