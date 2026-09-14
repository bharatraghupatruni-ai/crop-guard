from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.config import settings
from app.database import init_db
from app.routers import auth, users, scans, treatments, weather, analytics, admin, notifications, diagnose, research_feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CropGuard AI v3] Starting (MongoDB)...")
    await init_db()
    # Pre-warm ChromaDB RAG vector index in the background (non-blocking)
    from app.services.rag import warm_up as rag_warm_up
    rag_warm_up()
    yield
    print("[CropGuard AI] Shutting down")


app = FastAPI(
    title="CropGuard AI API",
    description="AI Crop Disease Detection — MongoDB + Visakhapatnam",
    version="3.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=settings.origins_list,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(auth.router,       prefix="/api/v1")
app.include_router(users.router,      prefix="/api/v1")
app.include_router(scans.router,      prefix="/api/v1")
app.include_router(treatments.router, prefix="/api/v1")
app.include_router(weather.router,    prefix="/api/v1")
app.include_router(analytics.router,  prefix="/api/v1")
app.include_router(admin.router,      prefix="/api/v1")
app.include_router(notifications.router,    prefix="/api/v1")
app.include_router(diagnose.router,         prefix="/api/v1")
app.include_router(research_feedback.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status":"ok","version":"3.0.0","db":"MongoDB","city":"Visakhapatnam"}

@app.get("/")
def root():
    return {"message":"CropGuard AI API v3","docs":"/docs"}
