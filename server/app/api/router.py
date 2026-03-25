from fastapi import APIRouter

from .routes.auth import router as auth_router
from .routes.devices import router as devices_router
from .routes.feedback import router as feedback_router
from .routes.health import router as health_router
from .routes.profile import router as profile_router
from .routes.sessions import router as sessions_router

api_router = APIRouter()
api_router.include_router(auth_router, tags=["Auth"])
api_router.include_router(devices_router, tags=["Devices"])
api_router.include_router(profile_router, tags=["Profile"])
api_router.include_router(feedback_router, tags=["Feedback"])
api_router.include_router(sessions_router, tags=["Sessions"])
api_router.include_router(health_router, tags=["Health"])
