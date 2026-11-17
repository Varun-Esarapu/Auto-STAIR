# Auto-STAIR — Workspace Rules & Hardware Constraints

## 1. Mission
Build **Auto-STAIR**, a CPU-friendly, zero-shot **Structure-Aware Retrieval & Multi-Hop RAG Engine** implementing Section 7 (Future Work) and Section 8 (Limitations) of IBM Research's STAIR paper (`arXiv:2609.03874`).

## 2. Strict Hardware & Dependency Guardrails (Windows i5 / 8GB RAM / No GPU)
- **NEVER** install or import `torch`, `transformers`, `bitsandbytes`, `peft`, or `sentence-transformers` in local code.
- All LLM reasoning (Dynamic ToC Induction, Zero-Shot Leaf Routing, Sufficiency Grading, and Final Answer Synthesis) MUST run via HTTP calls to the **Groq API** (`groq` Python SDK using `llama-3.1-8b-instant` for fast routing/induction and `llama-3.3-70b-versatile` for multi-hop synthesis) reading `GROQ_API_KEY` from `.env`.
- Keep local memory footprint under 500 MB. Stream Hugging Face datasets (`streaming=True`) instead of downloading full tables into RAM.

## 3. STAIR Architectural Standards
- **No Arbitrary Chunking:** Never slice PDFs into fixed 500-token chunks. Partition documents strictly along terminal Table-of-Contents leaf boundaries ($LN_D$).
- **Zero Structural Hallucinations:** Every router prediction MUST be validated against the document's valid leaf set $LN_D$ using our `TokenTrie` / enum constraint so invalid section headers cannot be emitted.
- **UI Framework:** Build the interactive frontend in **Gradio** (`app.py`) so it deploys directly onto Hugging Face Spaces' free `CPU Basic` tier.

## 4. Antigravity Execution & Verification Loop
- Always generate an `implementation_plan.md` in Planning Mode before creating or editing multi-file modules.
- Every module in `src/` requires unit tests in `tests/` using mocked Groq responses (so `pytest` runs offline in <3 seconds).
- Always run `pytest` in the terminal and fix any failures before finishing a task.
