"""Entity index for document-to-document matching.

Pre-computes a spaCy NER entity inverted index across all documents.
Supports fast entity→doc_id lookup and shared-entity discovery between doc pairs.

Usage:
    nlp = spacy.load("en_core_web_lg")
    docs = {doc_id: text for doc_id, text in ...}
    index = EntityIndex(nlp, docs)

    # Find web docs containing "ACC II"
    matching = index.docs_containing_text("ACC II", pool=web_doc_ids)

    # Find shared entities between two docs
    shared = index.shared_entities("local/DR0011/...", "web/drbench_urls/...")
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# spaCy has a processing limit; truncate very long docs
_SPACY_CHAR_LIMIT = 100_000

# Skip trivially short or purely numeric entities
_MIN_ENTITY_LEN = 2


def is_entity(ans: str) -> bool:
    """Check if a string looks like a useful bridge entity.

    Keeps: "Walmart", "25%", "$1.2 million", "2026", "Gen Z", "FDA"
    Rejects: "the year", "over half", "daily", "first"
    """
    ans = ans.strip()
    if not ans or len(ans) < 2:
        return False
    if len(ans.split()) > 2:
        return False
    if "@" in ans:
        return False
    # Must have at least one uppercase letter, digit, or $/%
    if not any(c.isupper() or c.isdigit() or c in "$%" for c in ans):
        return False
    return True


def _extract_entities(nlp, text: str) -> list[str]:
    """Extract unique entity strings from text using spaCy NER."""
    doc = nlp(text[:_SPACY_CHAR_LIMIT])
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        clean = ent.text.strip()
        if "\n" in clean or "\r" in clean:
            continue
        key = clean.lower()
        if len(key) < _MIN_ENTITY_LEN or key in seen:
            continue
        seen.add(key)
        entities.append(clean)
    return entities


class EntityIndex:
    """Pre-computed entity inverted index for fast entity→doc lookup.

    Built once at startup from all local + web documents.
    Uses spaCy en_core_web_lg for NER, with exact substring fallback.
    """

    def __init__(self, nlp, docs: dict[str, str]) -> None:
        """Build the index.

        Args:
            nlp: Loaded spaCy model (e.g., spacy.load("en_core_web_lg")).
            docs: {doc_id: full_text} for all documents to index.
        """
        t0 = time.time()
        self._nlp = nlp  # Keep for lazy NER on uncached docs
        self._lazy_lock = threading.Lock()

        # entity_text_lower -> set of doc_ids that contain this entity (per NER)
        self._entity_to_docs: dict[str, set[str]] = {}

        # doc_id -> list of entity strings (original case, from NER)
        self._doc_entities: dict[str, list[str]] = {}

        # doc_id -> full text (for substring fallback)
        self._doc_texts: dict[str, str] = {}

        # doc_id -> lowered text (for fast substring search)
        self._doc_texts_lower: dict[str, str] = {}

        for doc_id, text in docs.items():
            self._doc_texts[doc_id] = text
            self._doc_texts_lower[doc_id] = text.lower()

            entities = _extract_entities(nlp, text)
            self._doc_entities[doc_id] = entities
            for ent in entities:
                key = ent.lower()
                self._entity_to_docs.setdefault(key, set()).add(doc_id)

        elapsed = time.time() - t0
        n_ents = len(self._entity_to_docs)
        logger.info(
            "EntityIndex built: %d docs, %d unique entities, %.1fs",
            len(docs), n_ents, elapsed,
        )

    @property
    def doc_ids(self) -> set[str]:
        return set(self._doc_texts.keys())

    def entities_in_doc(self, doc_id: str) -> list[str]:
        """All spaCy entities extracted from a document (original case).

        Lazily runs NER for docs not in the pre-computed cache (thread-safe).
        """
        if doc_id in self._doc_entities:
            return self._doc_entities[doc_id]
        # Lazy NER for uncached docs (e.g., BrowseComp web docs)
        text = self._doc_texts.get(doc_id)
        if text is None or self._nlp is None:
            return []
        with self._lazy_lock:
            # Double-check after acquiring lock
            if doc_id in self._doc_entities:
                return self._doc_entities[doc_id]
            entities = _extract_entities(self._nlp, text)
            self._doc_entities[doc_id] = entities
            for ent in entities:
                self._entity_to_docs.setdefault(ent.lower(), set()).add(doc_id)
        return entities

    def docs_containing_entity_ner(self, entity: str) -> set[str]:
        """Find doc_ids where spaCy NER extracted this entity."""
        return set(self._entity_to_docs.get(entity.lower(), set()))

    def docs_containing_text(self, text: str, pool: set[str] | None = None) -> set[str]:
        """Find doc_ids whose text contains this string (exact, case-insensitive).

        This is the primary document selection method. More reliable than NER
        alone because spaCy misses domain-specific terms like "ACC II".
        """
        needle = text.strip().lower()
        if not needle:
            return set()
        result: set[str] = set()
        search_in = pool if pool is not None else self._doc_texts_lower.keys()
        for doc_id in search_in:
            doc_lower = self._doc_texts_lower.get(doc_id)
            if doc_lower is not None and needle in doc_lower:
                result.add(doc_id)
        return result

    def shared_entities(self, doc_id_a: str, doc_id_b: str) -> list[str]:
        """Entities appearing in both documents (via NER).

        Returns list of entity strings (original case from doc_a).
        Sorted by length descending (prefer more specific entities).
        """
        ents_a = {e.lower(): e for e in self._doc_entities.get(doc_id_a, [])}
        ents_b = {e.lower() for e in self._doc_entities.get(doc_id_b, [])}
        shared = [ents_a[k] for k in ents_a if k in ents_b]
        shared.sort(key=len, reverse=True)
        return shared

    def doc_text(self, doc_id: str) -> str:
        """Get full text for a document. Raises KeyError if not found."""
        return self._doc_texts[doc_id]

    def save(self, path: str | Path) -> None:
        """Save NER results to JSON so spaCy doesn't need to re-run."""
        data = {
            "doc_entities": self._doc_entities,  # {doc_id: [entity_str, ...]}
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False))
        logger.info("EntityIndex saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, docs: dict[str, str], nlp=None) -> "EntityIndex":
        """Load pre-computed NER from JSON, rebuild text indexes from docs.

        If nlp is provided, uncached docs get lazy NER when entities_in_doc is called.
        """
        data = json.loads(Path(path).read_text())
        obj = cls.__new__(cls)
        obj._nlp = nlp
        obj._lazy_lock = threading.Lock()
        obj._doc_entities = data["doc_entities"]
        obj._doc_texts = {}
        obj._doc_texts_lower = {}
        obj._entity_to_docs = {}

        for doc_id, text in docs.items():
            obj._doc_texts[doc_id] = text
            obj._doc_texts_lower[doc_id] = text.lower()

        for doc_id, entities in obj._doc_entities.items():
            for ent in entities:
                obj._entity_to_docs.setdefault(ent.lower(), set()).add(doc_id)

        n_cached = len(obj._doc_entities)
        n_total = len(docs)
        logger.info(
            "EntityIndex loaded: %d docs (%d cached, %d lazy), %d unique entities",
            n_total, n_cached, n_total - n_cached, len(obj._entity_to_docs),
        )
        return obj
