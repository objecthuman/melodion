import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from structlog import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import Logger, setup_logging
from src.vector_store import generate_and_upsert_embeddings, get_similar_tracks
from src.utils import get_audio_files
from src.middleware import LoggerMiddleWare


class EmbeddingRequest(BaseModel):
    file_paths: list[str] = Field(default=[])
    folder_path: str | None = Field(default=None)
    batch_size: int = Field(default=32, gt=0, le=128)


class EmbeddingResponse(BaseModel):
    success: bool
    count: int
    message: str


class SimilarityRequest(BaseModel):
    file_path: str
    top_k: int = Field(default=20, gt=0, le=100)


class SimilarTrack(BaseModel):
    id: str
    metadata: dict
    distance: float


class SimilarityResponse(BaseModel):
    query_file: str
    results: list[SimilarTrack]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


async def worker_process_manager(logger: Logger):
    while True:
        try:
            logger.info("Starting worker process", scan_interval=settings.SCAN_INTERVAL)

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "src.worker",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("Worker process completed successfully")
            else:
                logger.error(
                    "Worker process failed",
                    return_code=process.returncode,
                    stderr=stderr.decode() if stderr else None,
                )

        except Exception as e:
            logger.error("Error running worker process", error=str(e), exc_info=True)

        logger.info("Waiting for next scan", interval=settings.SCAN_INTERVAL)
        await asyncio.sleep(settings.SCAN_INTERVAL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging("melodion.log")
    logger: Logger = get_logger()
    logger.info("Music recommender started.")

    worker_task = asyncio.create_task(worker_process_manager(logger))
    logger.info("Worker process manager started")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Worker process manager cancelled")
        pass


app = FastAPI(
    title="Melodion (Music Recommendation API)", version="0.1.0", lifespan=lifespan
)

app.add_middleware(LoggerMiddleWare)


@app.get("/v1/health", response_model=HealthResponse)
async def health_check(logger: Logger):
    logger.info("Health check", model_loaded=True)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.post("/v1/music/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest, logger: Logger):
    if not Path(request.file_path).exists():
        logger.warning("File not found", file_path=request.file_path)
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )

    try:
        logger.info(
            "Finding similar tracks", file_path=request.file_path, top_k=request.top_k
        )

        results = get_similar_tracks(request.file_path, n_results=request.top_k)

        logger.info(
            "Similar tracks found",
            query_file=request.file_path,
            count=len(results),
        )

        similar_tracks = [
            SimilarTrack(
                id=results["ids"][0][i],
                metadata=results["metadatas"][0][i],
                distance=results["distances"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]

        return SimilarityResponse(
            query_file=request.file_path,
            results=similar_tracks,
            count=len(similar_tracks),
        )

    except Exception as e:
        logger.error("Error finding similar tracks", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar tracks: {str(e)}",
        )
