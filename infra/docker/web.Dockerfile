# Multi-stage Dockerfile for Next.js Studio Frontend
FROM node:24-alpine AS base

WORKDIR /app
RUN apk add --no-cache libc6-compat

# Install dependencies
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm ci

# Copy web app source
COPY apps/web .

ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production \
    PORT=3000

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
