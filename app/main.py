from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.routers.document_upload_route import router as document_router
from app.routers.query_route import router as query_router
from app.routers.summarize_route import router as summarize_router
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.config.mongodb_config import initialize_mongodb
from app.services.llm_service import list_models, default_choice
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_mongodb()
    yield


app = FastAPI(title="Document Analyzer API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

root_router = APIRouter()


@root_router.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@root_router.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}


@root_router.get("/api/v1/models")
async def get_models():
    """List available LLM providers and models for the frontend."""
    choice = default_choice()
    return {
        "providers": list_models(),
        "default": {"provider": choice[0], "model": choice[1]} if choice else None,
    }


templates = Jinja2Templates(directory="app/templates")


@app.get("/api/v1", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(summarize_router, prefix="/api/v1", tags=["Summarize"])
app.include_router(root_router, tags=["Root"])
