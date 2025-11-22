from pathlib import Path
from typing import TypedDict

import chromadb
import torch
from chromadb.config import Settings
from laion_clap import CLAP_Module as ClapModel
from mutagen._file import File as MutagenFile


class MusicMetadata(TypedDict):
    artist: str
    genre: str
    song_name: str
    album_name: str
    file_path: str


class EmbeddingResult(TypedDict):
    metadatas: list[MusicMetadata]
    ids: list[str]


class SimilarityResult(TypedDict):
    ids: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]


ChromaClient = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(
        is_persistent=True,
    ),
)


Collection = ChromaClient.get_or_create_collection(
    name="udio_embeddings",
    metadata={"hnsw:space": "cosine"},
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

clap_model = ClapModel(enable_fusion=False, device=device)
clap_model.load_ckpt()


def extract_metadata(file_path: str) -> MusicMetadata:
    try:
        audio = MutagenFile(file_path)
        if audio is None:
            return MusicMetadata(
                artist="Unknown",
                genre="Unknown",
                song_name=Path(file_path).stem,
                album_name="Unknown",
                file_path=file_path,
            )

        artist = "Unknown"
        genre = "Unknown"
        song_name = Path(file_path).stem
        album_name = "Unknown"

        if audio.tags:
            artist = (
                str(audio.tags.get("TPE1", ["Unknown"])[0])
                if "TPE1" in audio.tags
                else str(audio.tags.get("artist", ["Unknown"])[0])
                if "artist" in audio.tags
                else "Unknown"
            )
            genre = (
                str(audio.tags.get("TCON", ["Unknown"])[0])
                if "TCON" in audio.tags
                else str(audio.tags.get("genre", ["Unknown"])[0])
                if "genre" in audio.tags
                else "Unknown"
            )
            song_name = (
                str(audio.tags.get("TIT2", [Path(file_path).stem])[0])
                if "TIT2" in audio.tags
                else str(audio.tags.get("title", [Path(file_path).stem])[0])
                if "title" in audio.tags
                else Path(file_path).stem
            )
            album_name = (
                str(audio.tags.get("TALB", ["Unknown"])[0])
                if "TALB" in audio.tags
                else str(audio.tags.get("album", ["Unknown"])[0])
                if "album" in audio.tags
                else "Unknown"
            )

        return MusicMetadata(
            artist=artist,
            genre=genre,
            song_name=song_name,
            album_name=album_name,
            file_path=file_path,
        )
    except Exception:
        return MusicMetadata(
            artist="Unknown",
            genre="Unknown",
            song_name=Path(file_path).stem,
            album_name="Unknown",
            file_path=file_path,
        )


def generate_and_upsert_embeddings(
    file_paths: list[str], batch_size: int = 32
) -> EmbeddingResult:
    all_metadatas = []
    all_ids = []

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i : i + batch_size]

        batch_metadatas = []
        batch_ids = []
        for file_path in batch_paths:
            metadata = extract_metadata(file_path)
            batch_metadatas.append(metadata)

            id_str = f"{metadata['artist']}_{metadata['song_name']}_{metadata['album_name']}_{metadata['genre']}_{Path(file_path).stem}".replace(
                " ", "_"
            )
            batch_ids.append(id_str)

        batch_embeddings = clap_model.get_audio_embedding_from_filelist(
            x=batch_paths,
            use_tensor=False,
        )

        batch_embeddings_list = batch_embeddings.tolist()

        Collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings_list,
            metadatas=batch_metadatas,
        )

        all_metadatas.extend(batch_metadatas)
        all_ids.extend(batch_ids)

    return EmbeddingResult(
        metadatas=all_metadatas,
        ids=all_ids,
    )


def get_similar_tracks(file_path: str, n_results: int = 20) -> SimilarityResult:
    embedding = clap_model.get_audio_embedding_from_filelist(
        x=[file_path],
        use_tensor=False,
    )

    embedding_list = embedding.tolist()[0]

    results = Collection.query(
        query_embeddings=[embedding_list],
        n_results=n_results,
    )

    return results
