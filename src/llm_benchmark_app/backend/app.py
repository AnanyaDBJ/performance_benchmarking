from .core import create_app
from .router import router
from .benchmark_router import benchmark_router

app = create_app(routers=[router, benchmark_router])
