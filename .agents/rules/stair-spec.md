---
trigger: manual
---

# STAIR Technical Specification (arXiv:2609.03874v2)

## 1. Formal Tree & Retrieval Definitions (Section 4.1)
- Document $D$ has a Table of Contents tree $ToC_D = \{T_1, T_2, \dots, T_n\}$.
- Directed edge $e: T_p \rightarrow T_c$ denotes $T_p = \text{parent}(T_c)$.
- Leaf node set $LN_D \subseteq ToC_D$ contains terminal section headers with zero children.
- Goal: Given user query $q$ and global structure $ToC_D$, predict the exact leaf node(s) $T_l \in LN_D$ whose intact section text answers $q$.

## 2. Exact Table 2 Prompt Format
```text
### Book: {book_title}
Pick the best section from the table of contents below which can answer the user query below. Only generate the section name, do not generate any explanation!

### Table of Contents
{formatted_toc_with_leaf_prefixes}

### Query: {query}

### Section: leaf
```

## 3. Official meghanadh/SearchTome Dataset Schema
Repo ID: `meghanadh/SearchTome` (CC BY-SA 4.0)
Domains (6): `education`, `finance`, `law`, `medicine`, `natural_sciences`, `social_sciences`
Configs per Domain:
- `{domain}_corpus` (split: `corpus`): `{"title": str, "content": str}`
- `{domain}_queries` (splits: `train`, `dev`, `test`): `{"query_id": int, "query": str, "gold_doc_title": str, "answer": str}`

## 4. Paper Baselines to Benchmark Against (Table 4)
- Zero-Shot Mistral Instruct v0.2: 13.8% Recall@1 | 16.4% Recall@3 | 15.4% nDCG@3
- BM25 (Lexical Baseline): 59.5% Recall@1 | 77.6% Recall@3 | 70.1% nDCG@3
- Fine-Tuned STAIR (Paper Upper Bound): 82.6% Recall@1 | 90.8% Recall@3 | 87.5% nDCG@3 | <0.05% hallucination rate
