# AI Proxy

A lightweight HTTP proxy server for routing requests to multiple OpenAI-compatible API services with automatic health checking and load balancing.

## Overview

This proxy acts as a unified entry point for AI model APIs, allowing you to:

- Route requests to multiple backend services based on URL paths
- Automatically select healthy services for request forwarding
- Support model aliasing for flexible routing
- Aggregate model listings from multiple backends into a single view
- Bridge Gemini-specific embedding endpoints to OpenAI-compatible formats

## Features

- **Service Routing**: Forward requests to configured backend services based on route patterns
- **Health Checking**: Automatic health checks with caching to ensure only healthy services receive traffic
- **Model Aggregation**: Combine model listings from multiple backends into a unified `/v1/models` endpoint
- **Gemini Embedding Bridge**: Convert Gemini-specific embedding endpoints to OpenAI-compatible format
- **Plugin Architecture**: Extensible design allowing custom request handling via plugins

## Configuration

Services and routing rules are defined in `config.yaml`:

```yaml
services:
  - name: Service Name
    baseUrl: http://host:port
    token: your-api-token-or-not-needed
    health: /v1/models
    routes:
      - /v1/chat/completion
      - /v1/models
```

## Deployment

Build and run with Docker:

```bash
docker build -t ai-proxy .
docker run -p 8080:8080 ai-proxy
```

Or use docker-compose for volume mounting of config.yaml.

## Usage

The proxy listens on port 8080 and forwards requests to configured services based on route matching in the request path.
