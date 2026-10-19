from fastapi import APIRouter

from app.api.v1 import experiments, human_eval, pipeline, system

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router, tags=["system"])
api_router.include_router(pipeline.router, tags=["pipeline"])
api_router.include_router(experiments.router, tags=["experiments"])
api_router.include_router(human_eval.router, tags=["human-eval"])
