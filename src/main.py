from pathlib import Path
from contextlib import asynccontextmanager

from structlog import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from logger import Logger, setup_logging
from vector_store import generate_and_upsert_embeddings, get_similar_tracks


class EmbeddingRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging("melodion.log")
    logger = get_logger()
    logger.info("Music recommender started.")

    yield


app = FastAPI(
    title="Melodion (Music Recommendation API)", version="0.1.0", lifespan=lifespan
)


@app.get("/v1/health", response_model=HealthResponse)
async def health_check(logger: Logger):
    logger.info("Health check", model_loaded=True)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.post("/v1/music/index", response_model=EmbeddingResponse)
async def index_music(request: EmbeddingRequest, logger: Logger):
    invalid_files = []
    for file_path in request.file_paths:
        if not Path(file_path).exists():
            invalid_files.append(file_path)

    if invalid_files:
        logger.warning("Invalid files", files=invalid_files)
        raise HTTPException(
            status_code=400,
            detail=f"Files not found: {', '.join(invalid_files)}",
        )

    try:
        logger.info(
            "Indexing music files",
            file_count=len(request.file_paths),
            batch_size=request.batch_size,
        )

        result = generate_and_upsert_embeddings(
            request.file_paths, batch_size=request.batch_size
        )

        logger.info(
            "Music files indexed successfully",
            count=len(result["ids"]),
        )
        return EmbeddingResponse(
            success=True,
            count=len(result["ids"]),
            message=f"Successfully indexed {len(result['ids'])} music files",
        )

    except Exception as e:
        logger.error("Error indexing music files", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing music files: {str(e)}",
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
