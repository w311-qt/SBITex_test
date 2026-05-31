from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    file_path: str  # relative to repo root, e.g. "Src/DiffWrapper.cpp"
    start_line: int  # 1-based
    end_line: int
    text: str


def _chunk_file(
    path: Path,
    repo_root: Path,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    rel = path.relative_to(repo_root).as_posix()
    step = max(1, chunk_size - overlap)
    n = len(lines)
    chunks: list[Chunk] = []

    for start in range(0, max(n, 1), step):
        end = min(start + chunk_size, n)
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(Chunk(file_path=rel, start_line=start + 1, end_line=end, text=text))
        if end >= n:
            break

    return chunks


def build_chunks(
    repo_root: Path,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    src = repo_root / "Src"
    chunks: list[Chunk] = []
    for pattern in ("**/*.cpp", "**/*.h"):
        for path in sorted(src.glob(pattern)):
            chunks.extend(_chunk_file(path, repo_root, chunk_size, overlap))
    return chunks
