# Developer Onboarding Guide

Welcome to the Interview Analyzer project! This guide will help you set up your development environment from scratch on macOS using Cursor IDE.

## Overview

This project is a sophisticated event-sourced application for analyzing interview transcripts using AI. It features:

- FastAPI-based REST API
- Event-sourced architecture with EventStoreDB
- Neo4j graph database for projections
- Celery workers for background processing
- Real-time projection service (sole Neo4j writer) + live SSE UI feed
- A layered "Mine" analysis pipeline (structure → enrichment → lenses →
  segments → export/ask) and a two-surface Next.js UI
- Comprehensive test suite (current counts in [../ROADMAP.md](../ROADMAP.md))

## Onboarding Path

Follow these guides in order. Each guide builds on the previous one.

### 1. [Prerequisites](./01-prerequisites.md) ⏱️ ~30 minutes

Set up your Mac development environment with required tools:

- Homebrew package manager
- Docker Desktop
- Cursor IDE
- Git configuration

### 2. [Initial Setup](./02-initial-setup.md) ⏱️ ~15 minutes

Clone the repository and configure the project:

- Clone repository
- Create `.env` file with API keys
- Build Docker images
- Verify build

### 3. [Running the System](./03-running-the-system.md) ⏱️ ~20 minutes

Start all services and verify they're working:

- Start 7 Docker services
- Access web interfaces
- Run first pipeline test
- Verify data flow

### 4. [Architecture Overview](./04-architecture-overview.md) ⏱️ ~30 minutes (reading)

Understand the system architecture:

- Service descriptions
- Event-sourced architecture
- Data flow
- Directory structure
- Configuration system

### 5. [Development Workflow](./05-development-workflow.md) ⏱️ ~20 minutes

Learn daily development tasks:

- Running tests
- Using Makefile targets
- Managing databases
- API development
- Debugging

### 6. [Troubleshooting](./06-troubleshooting.md) 📚 Reference

Common issues and solutions (consult as needed)

### 7. [Hidden Files and Cursor Setup](./07-hidden-files-and-cursor.md) 📚 Reference

Configuration files explained: `.env`, `.cursorrules`, `.gitignore`, Docker files

### 8. [DevContainer Setup](./08-devcontainer-setup.md) 📚 Reference

DevContainer configuration for Cursor IDE (automatic environment setup)

### ⚠️ [Security Warning](./SECURITY-WARNING.md) 🚨 CRITICAL

**READ THIS:** Information about API key exposure and security best practices

### 📁 [File Locations Update](./FILE_LOCATIONS_UPDATE.md) 📋 Reference

Configuration template files locations and usage (env.example, cursorrules.example, devcontainer.env.example)

## Quick Reference

### Essential Commands

```bash
# Start all services
docker compose up -d

# Run tests
make test

# Run linting
make lint

# Stop services
docker compose down

# Clean everything (including data)
docker compose down -v
```

### Service URLs

- API Documentation: http://localhost:8000/docs
- Neo4j Browser (Main): http://localhost:7474
- Neo4j Browser (Test): http://localhost:7475
- EventStoreDB UI: http://localhost:2113

### Getting Help

- Check the [Troubleshooting Guide](./06-troubleshooting.md)
- Review service logs: `docker compose logs <service-name>`
- Contact your team lead for specific communication channels
- Provide diagnostic info when asking for help (see troubleshooting guide)

## Total Estimated Setup Time

**~2 hours** (including breaks and reading)

Ready to begin? Start with [Prerequisites →](./01-prerequisites.md)
