"""Text encoders and the NLI scorer used by the experiments.

Every encoder returns L2-normalized vectors, so a dot product is a cosine similarity.
Backends:
  sentence_transformers  - pooling as defined by the model card (mpnet, LaBSE, e5)
  hf_meanpool            - transformers AutoModel, mean over real tokens (padding masked)
  hf_meanpool_nomask     - mean over all positions including padding (the reported run)
  hashing                - character n-gram hashing; no model download, for tests only
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _device(name: str = "auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class Encoder:
    """Common interface: encode(texts, kind) with kind in {"query", "passage"}."""

    def __init__(self, name, backend, query_prefix="", passage_prefix="",
                 batch_size=64, device="auto"):
        self.name, self.backend = name, backend
        self.prefix = {"query": query_prefix, "passage": passage_prefix}
        self.batch_size = batch_size
        self._cache: dict = {}
        if backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name, device=str(_device(device)))
        elif backend in ("hf_meanpool", "hf_meanpool_nomask"):
            from transformers import AutoModel, AutoTokenizer
            self.dev = _device(device)
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name).to(self.dev).eval()
        elif backend != "hashing":
            raise ValueError(f"unknown backend {backend}")

    @property
    def label(self) -> str:
        return f"{self.name} [{self.backend}]"

    def encode(self, texts, kind="passage") -> np.ndarray:
        texts = [self.prefix[kind] + str(t) for t in texts]
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._encode(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def tokenizer(self):
        if self.backend == "sentence_transformers":
            return self.model.tokenizer
        return getattr(self, "tok", None)

    def encode_chunked(self, texts, kind="passage", size=120, overlap=20) -> list:
        """One (n_chunks, dim) array per text; the encoder's max_seq_length truncates longer inputs."""
        chunks = [split_into_chunks(str(t), size, overlap, self.tokenizer()) for t in texts]
        flat = self.encode([c for cs in chunks for c in cs], kind)
        out, i = [], 0
        for cs in chunks:
            out.append(flat[i:i + len(cs)])
            i += len(cs)
        return out

    def _encode(self, texts):
        if self.backend == "sentence_transformers":
            return _normalize(self.model.encode(texts, batch_size=self.batch_size,
                                                normalize_embeddings=True, show_progress_bar=False))
        if self.backend == "hashing":
            return _normalize(np.stack([_hash_vec(t) for t in texts]))
        import torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            if self.backend == "hf_meanpool_nomask":
                v = h.mean(dim=1)
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(v.float().cpu().numpy())
        return _normalize(np.vstack(out))


def split_into_chunks(text: str, size=120, overlap=20, tokenizer=None) -> list:
    """Windows of `size` tokens with `overlap` tokens shared, cut from the original text by
    character offsets (whitespace tokens when no tokenizer is given)."""
    if tokenizer is None:
        spans = [m.span() for m in re.finditer(r"\S+", text)]
    else:
        spans = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                          verbose=False)["offset_mapping"]
    if len(spans) <= size:
        return [text]
    step, out = size - overlap, []
    for i in range(0, len(spans), step):
        w = spans[i:i + size]
        out.append(text[w[0][0]:w[-1][1]])
        if i + size >= len(spans):
            break
    return out


def _hash_vec(text: str, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    t = f"  {text.lower()}  "
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            h = int(hashlib.md5(t[i:i + n].encode()).hexdigest(), 16)
            v[h % dim] += 1.0
    return v


def make_encoder(spec: dict, batch_size=64, device="auto") -> Encoder:
    return Encoder(spec["name"], spec["backend"], spec.get("query_prefix", ""),
                   spec.get("passage_prefix", ""), batch_size, device)


class NLIScorer:
    """Entailment probability P(entailment | premise, hypothesis) from a softmax over the
    three NLI classes. backend="lexical" is a token-overlap stand-in for tests only."""

    def __init__(self, model_name, batch_size=64, device="auto", backend="hf"):
        self.backend, self.batch_size = backend, batch_size
        if backend == "hf":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.dev = _device(device)
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.dev).eval()
            l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
            if "entailment" not in l2i:
                raise ValueError(f"no entailment label in {self.model.config.label2id}")
            self.ent = l2i["entailment"]

    def entail(self, premises, hypotheses) -> np.ndarray:
        if self.backend == "lexical":
            tok = lambda s: set(re.findall(r"\w+", s.lower()))
            return np.array([len(tok(h) & tok(p)) / max(len(tok(h)), 1)
                             for p, h in zip(premises, hypotheses)], dtype=np.float32)
        import torch
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                           padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.dev)
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.ent]
            out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)
