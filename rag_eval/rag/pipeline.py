from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from .indexer import BM25Index
from ..config import Settings

MAX_CONTEXT_CHARS = 12_000

_SYSTEM_PROMPT = """\
You are an expert on the WinMerge C++ codebase (a Windows file/folder comparison tool built with MFC/Win32).
Answer the question using ONLY the provided source code context.
If the answer cannot be determined from the context, say clearly: "This information is not present in the retrieved context."
Be concise and precise; reference specific class/function names and file paths.
"""

_USER_TEMPLATE = """\
## Retrieved source code context

{context}

## Question

{question}
"""

_MOCK_ANSWER = "[DRY-RUN] Mock answer — real LLM call skipped."


@dataclass
class RagAnswer:
    answer: str
    context: str        # exact text sent to the generator
    evidence_files: list[str]  # unique file paths from retrieval


AnswerFn = Callable[[str], RagAnswer]


def _build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[str]]:
    """Build context string and collect all retrieved file paths.

    evidence_files = all unique files from BM25 retrieval (for evidence_recall metric).
    Context is built in BM25 rank order and capped at max_chars for the generator prompt.
    A file counts as retrieved even if its chunk was too large to fit in the context.
    """
    parts: list[str] = []
    context_full = False
    all_files: list[str] = []
    all_seen: set[str] = set()
    total = 0

    for r in results:
        # Track all retrieved files regardless of context capacity
        if r.chunk.file_path not in all_seen:
            all_seen.add(r.chunk.file_path)
            all_files.append(r.chunk.file_path)

        if not context_full:
            header = f"// === {r.chunk.file_path}  L{r.chunk.start_line}-{r.chunk.end_line} ===\n"
            snippet = header + r.chunk.text
            if total + len(snippet) > max_chars:
                context_full = True  # stop adding to context but keep tracking files
            else:
                parts.append(snippet)
                total += len(snippet)

    return "\n\n".join(parts), all_files


def make_answer_fn(settings: Settings, index: BM25Index, top_k: int = 5) -> AnswerFn:
    client = OpenAI(base_url=settings.model_base_url, api_key=settings.model_api_key or "none")

    def answer_fn(question: str) -> RagAnswer:
        results = index.search(question, top_k=top_k)
        context, evidence_files = _build_context(results)

        if settings.dry_run:
            return RagAnswer(answer=_MOCK_ANSWER, context=context, evidence_files=evidence_files)

        resp = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(context=context, question=question)},
            ],
            temperature=0.1,
            max_tokens=settings.max_tokens_generator,
            timeout=settings.request_timeout_sec,
            extra_body={"options": {"num_ctx": settings.num_ctx}},
        )
        return RagAnswer(
            answer=resp.choices[0].message.content or "",
            context=context,
            evidence_files=evidence_files,
        )

    return answer_fn
