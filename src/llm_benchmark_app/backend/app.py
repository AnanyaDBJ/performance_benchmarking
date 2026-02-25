from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .benchmark.db import create_pool, init_tables, is_postgres_configured
from .core import create_app
from .router import router
from .benchmark_router import benchmark_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if is_postgres_configured():
        logger.info("Postgres detected — initializing connection pool")
        pool = create_pool(app.state.workspace_client)
        init_tables(pool)
        app.state.db_pool = pool
        logger.info("Database pool ready")
    else:
        logger.info("No PGHOST found — running with in-memory storage only")
        app.state.db_pool = None

    yield

    pool = getattr(app.state, "db_pool", None)
    if pool is not None:
        pool.close()
        logger.info("Database pool closed")


app = create_app(routers=[router, benchmark_router], lifespan=_lifespan)
