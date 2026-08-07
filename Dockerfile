FROM node:20-alpine AS frontend-build
WORKDIR /workspace/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM python:3.11-slim
WORKDIR /workspace/backend
COPY backend/requirements.txt backend/requirements-ml.txt backend/constraints.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt -c constraints.txt
COPY backend ./
COPY --from=frontend-build /workspace/frontend/dist /workspace/frontend-dist
ENV FRONTEND_DIST=/workspace/frontend-dist
EXPOSE 8080
CMD ["sh", "-c", "if [ \"$SERVICE_ROLE\" = \"worker\" ]; then exec celery -A app.worker.celery_app worker --loglevel=INFO --concurrency=1; elif [ \"$SERVICE_ROLE\" = \"scheduler\" ]; then exec celery -A app.worker.celery_app beat --loglevel=INFO; else alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}; fi"]
