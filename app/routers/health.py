from fastapi import APIRouter
from fastapi.responses import Response

from app.memory_util import rss_mb

router = APIRouter(tags=["health"])


@router.get("/")
def home():
    return {"message": "Kvali Engine is running"}


@router.head("/")
def home_head():
    return Response(status_code=200)


@router.get("/debug/memory")
def debug_memory():
    return {"rss_mb": round(rss_mb(), 2)}
