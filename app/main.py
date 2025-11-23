from fastapi import FastAPI, APIRouter
from routers.chat_init_route import router as chat_init_router
from routers.rag_route import router as rag_router
from routers.summarize_route import router as summarize_route
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
router = APIRouter()




@router.get("/")
async def read_root():
    return {"Hello": "World"}


app.include_router(chat_init_router, prefix= "/chat_init")
app.include_router(rag_router, prefix="/ask")
app.include_router(summarize_route, prefix="/summarize")
app.include_router(router)