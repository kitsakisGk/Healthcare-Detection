# Deployment Guide

This folder contains all deployment configurations for different platforms.

## Structure

```
deployment/
├── docker/              # Docker deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── huggingface/        # Hugging Face Spaces
│   ├── README.md
│   └── config.yaml
└── README.md           # This file
```

## Quick Deploy

### Docker

```bash
cd deployment/docker
docker-compose up -d
```

App runs at: http://localhost:8501

### Hugging Face Spaces

See [../DEPLOY_HF.md](../DEPLOY_HF.md) for complete guide.

Quick: Upload files via HF web interface.

## Production Checklist

- [ ] Update model files
- [ ] Configure environment variables
- [ ] Set up logging
- [ ] Enable HTTPS
- [ ] Configure backups
- [ ] Set resource limits
- [ ] Test on production data
