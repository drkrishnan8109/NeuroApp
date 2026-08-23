"""Clinician-facing front end for the neurology RAG.

Deployment shape (Streamlit Community Cloud):
  * queries are embedded through the HF Inference API with BAAI/bge-m3 — the same model
    that built the index, so the 1024-d vectors match;
  * generation runs on a hosted model (Ollama is unreachable from a container);
  * the BM25 arm is rebuilt from text already stored in Pinecone metadata, so the
    licensed PDFs never have to ship with the app.

Every answer passes through src.guardrails.guard(), which is what that module was written
for: input-injection blocking and PII/size caps on the way in, then HTML sanitisation,
PII redaction, a non-diagnostic disclaimer and citations on the way out.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # local dev

# On Community Cloud there is no .env — secrets come from the app's settings UI and arrive
# as st.secrets. Copy them into the environment so every os.getenv() in src/ keeps working
# unchanged, rather than threading a config object through the whole pipeline.
# Existing environment variables win, so a local .env still takes precedence.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass          # no secrets.toml configured: fine when running locally

from langchain.chat_models import init_chat_model

from src.guardrails import guard, GuardrailConfig
from src.rag_pipeline import Retriever, build_bm25_local, get_backend, make_rag_bot

st.set_page_config(page_title="Neurology Reference Assistant", page_icon="🧠", layout="centered")


# ----------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Connecting to the index…")
def _backend():
    """Preferred vector store, falling back to the local one when it is unreachable."""
    return get_backend()


@st.cache_resource(show_spinner="Building the keyword index…")
def _bm25(enabled: bool, chunk_size: int, chunk_overlap: int):
    """BM25 built from the local PDF cache — a couple of seconds, and no metered egress.

    Chunk size is passed in rather than assumed, because it must match the vector store in
    use: the two arms are fused by rank, and ranks over different chunk boundaries are not
    comparable.
    """
    if not enabled:
        return None
    return build_bm25_local(chunk_size, chunk_overlap)


@st.cache_resource(show_spinner="Starting the model…")
def _llm(model_id: str):
    return init_chat_model(model_id, temperature=0.0)


# Generator fallback chain, tried in order.
# Local first: qwen2.5:14b is the model every recorded score was produced with, and it has
# no quota or rate limit. The hosted entries cover the case where Ollama is unreachable (a
# deployed container cannot run it) and each other's free-tier 429/503s.
GENERATORS = [g.strip() for g in os.getenv(
    "RAG_GENERATORS",
    "ollama:qwen2.5:14b,mistralai:mistral-small-latest,google_genai:gemini-flash-latest",
).split(",") if g.strip()]


def _answer_with_fallback(retriever, mode, k, question):
    """Try each generator in turn; report which one answered."""
    errors = []
    for model_id in GENERATORS:
        try:
            bot = make_rag_bot(_llm(model_id), retriever, mode=mode, k=k)
            return guard(bot, GuardrailConfig())(question), model_id, errors
        except Exception as exc:
            errors.append(f"{model_id}: {type(exc).__name__} {str(exc)[:120]}")
    raise RuntimeError("All generators failed:\n" + "\n".join(errors))


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Retrieval")
    # hybrid/keyword are meaningless without the BM25 arm; warn rather than degrade silently
    mode = st.selectbox(
        "Mode", ["hybrid", "vector", "keyword"], index=0,
        help=("hybrid fuses both arms by rank (RRF). vector is semantic similarity only. "
              "keyword is BM25 exact-term matching, which scored highest on correctness "
              "in evaluation."),
    )
    k = st.slider("Passages to use", 3, 10, 6,
                  help="How many retrieved chunks are placed in the model's prompt.")
    use_bm25 = st.toggle(
        "Enable keyword arm", value=True,
        help=("Off by default because it must rebuild a BM25 index from the 21,030 chunks "
              "built from the local PDF cache in a couple of seconds. On by default because "
              "keyword retrieval scored highest on correctness in evaluation (0.90, against "
              "0.70 for vector and 0.60 for hybrid). Switch it off for instant startup and "
              "vector-only retrieval."),
    )
    st.divider()
    _b = _backend()
    st.caption(
        f"**Index:** `{_b.name}` — {_b.store.count():,} vectors, {_b.chunk_size}/{_b.chunk_overlap} chunks"
    )
    st.caption(
        f"**Answer floor:** cosine ≥ {_b.min_cosine:.2f}\n\n"
        "Questions whose best passage scores below this are declined rather than "
        "answered from weak evidence, calibrated against this index's own score "
        "distribution — the floor is not transferable between embedding models."
    )
    st.caption("**Generators (in order):** " + ", ".join(f"`{g}`" for g in GENERATORS))


# ---------------------------------------------------------------------------- main
st.title("Neurology Reference Assistant")
st.caption(
    "Answers are drawn only from four indexed references — Bradley's *Neurology in "
    "Clinical Practice*, *Clinical Neurophysiology*, Harrison's *Neurology in Clinical "
    "Medicine*, and the *Handbook of Emergency Neurology*. Questions outside them are "
    "declined rather than guessed at."
)

question = st.text_area(
    "Clinical question",
    placeholder="e.g. What is the management of GBS if there is no improvement after IVIG?",
    height=90,
)
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    if mode in ("hybrid", "keyword") and not use_bm25:
        st.warning(
            f"`{mode}` mode needs the keyword arm, which is switched off — enable it in the "
            "sidebar or switch to `vector` mode."
        )
        st.stop()
    try:
        b = _backend()
        retriever = Retriever.from_backend(b, _bm25(use_bm25, b.chunk_size, b.chunk_overlap))
        with st.spinner("Searching the references…"):
            result, used_model, tried = _answer_with_fallback(retriever, mode, k, question)
        note = f"`{_backend().name}` index · `{used_model}`"
        if tried:
            note += f" (after {len(tried)} generator(s) failed)"
        st.caption(note)
    except Exception as exc:  # surface the cause instead of a blank page
        st.error(f"Could not answer that: {exc}")
        st.stop()

    flags = (result.get("guardrails") or {})
    if flags.get("blocked"):
        st.warning("That request was blocked before reaching the model.")
        st.write(result["answer"])
        st.stop()

    if result.get("abstained"):
        st.info("**Not covered by the indexed references.**")
        st.write(result["answer"])
        st.caption(
            f"Best passage scored cosine {result['best_cosine']:.3f}, below the "
            f"{_backend().min_cosine:.2f} answer floor — the closest text in the corpus was not "
            "similar enough to answer from."
        )
        st.stop()

    st.markdown(result["answer"])

    docs = result.get("documents") or []
    if docs:
        st.divider()
        st.subheader("Sources")
        for i, d in enumerate(docs, 1):
            meta = d.get("metadata") or {}
            src = meta.get("source_file", "unknown")
            page = meta.get("page", "?")
            bits = []
            if d.get("similarity") is not None:
                bits.append(f"cosine {d['similarity']:.3f}")
            if d.get("bm25_score") is not None:
                bits.append(f"BM25 {d['bm25_score']:.2f}")
            if d.get("fusion_score") is not None:
                bits.append(f"RRF {d['fusion_score']:.4f}")
            with st.expander(f"{i}. {src} · p.{page}  —  {' · '.join(bits)}"):
                st.write(d["content"])

        with st.popover("How to read these scores"):
            st.markdown(
                "- **cosine** — semantic closeness, 0–1, comparable across questions. "
                "Roughly 1.0 means near-identical meaning.\n"
                "- **BM25** — exact-term match strength. Unbounded and corpus-relative, "
                "so it is only comparable *within* this list, never between questions.\n"
                "- **RRF** — the fusion score used to order this list. Built from rank "
                "position, not relevance, so its absolute value carries no meaning; a "
                "passage found by both arms scores roughly double one found by one."
            )

st.divider()
st.caption(
    "⚕️ Decision support only — not a diagnosis, and not a substitute for clinical "
    "judgement or current local guidelines. Verify against the cited page before acting."
)
