from fastapi import FastAPI, APIRouter
from routers.document_upload_route import router as document_router
from routers.query_route import router as query_router
from routers.summarize_route import router as summarize_router
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from config.mongodb_config import initialize_mongodb
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Code to run at server startup
    await initialize_mongodb()
    yield
    # Code to run at shutdown (optional)

app = FastAPI(
    title="Document Analyzer API",
    description="API for document processing, summarization, and querying using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Root router
root_router = APIRouter()

@root_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "documents": "/api/v1/documents",
            "query": "/api/v1/query",
            "summarize": "/api/v1/summarize"
        }
    }

@root_router.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

templates = Jinja2Templates(directory="app/templates")

# Include routers with API version prefix
@app.get("/api/v1", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(summarize_router, prefix="/api/v1", tags=["Summarize"])
app.include_router(root_router, tags=["Root"])
