from fastapi import FastAPI, UploadFile, HTTPException
from transcriber import Transcriber
from time import time as now
import logging

app = FastAPI(title="faster-whisper-fr-api")
logger = logging.getLogger("uvicorn")
model = None


@app.on_event("startup")
def on_startup():
    started = now()
    global model
    model = Transcriber()
    logger.info(f"MODEL: {model.model_path} - Loaded ({now() - started:.2f}s)")


@app.get("/")
async def ping():
    logger.info("API: Ping!")
    return {"status": "OK"}


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    text, stats = "", []
    started = now()
    logger.info(f"FILE: {file.filename} - Start processing...")
    try:
        text, stats = model.process(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error during transcription: " + str(e)
        )
    stop = now() - started
    logger.info(f"FILE: {file.filename} - Processed ({stop/60:.0f}m{stop%60:.2f}s)")
    return {"time": stop, "text": text, "stats": stats}
