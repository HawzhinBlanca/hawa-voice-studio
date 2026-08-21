# Production Deployment & Operations Runbook

## 1. Local Development
```bash
# Start PostgreSQL & backend services
docker-compose -f infra/docker/docker-compose.yml up -d

# Run FastAPI Control Plane
uvicorn services.api.main:app --reload --port 8000

# Run Next.js Studio Frontend
cd apps/web && npm run dev
```

## 2. Canary Deployment Workflow
1. Train model checkpoint and pass Evaluation Gate (`/v1/evaluations/{id}/approve`).
2. Deploy as Canary with 10% initial traffic (`POST /v1/deployments/{id}/canary`).
3. Monitor latency (P95 TTFB < 500ms) and user error rates.
4. Promote to 100% active production traffic (`POST /v1/deployments/{id}/promote`).

## 3. Disaster Recovery & Rollback
- If unexpected quality degradation or regression is detected, issue immediate rollback:
  `POST /v1/deployments/{id}/rollback`
- Traffic instantly reverts to the prior stable production foundation.
