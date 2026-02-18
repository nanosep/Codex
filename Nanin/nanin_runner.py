import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI


CATEGORY_PATTERNS = {
    "Artifacts": ["artifact"],
    "Bonds & Beasts": ["bond", "beast"],
    "Cards": ["card"],
    "Glamour": ["glamour"],
    "Rituals": ["ritual"],
    "Weapons": ["weapon"],
}

DEFAULT_CATEGORY_ORDER = [
    "Artifacts",
    "Bonds & Beasts",
    "Cards",
    "Glamour",
    "Rituals",
    "Weapons",
]

DOC_FIELDS = [
    "object",
    "type",
    "object_class",
    "council_objective",
    "when_to_use",
    "key_questions",
    "logic_narrative",
    "framework_lens",
    "technique_hooks",
    "prompting_techniques",
    "acceptance_criteria",
    "expected_outputs",
]


def flatten(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten(item))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for k, v in value.items():
            out.append(str(k))
            out.extend(flatten(v))
        return out
    return [str(value)]


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def read_json_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(text)


def normalize_objects(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["ETHOS", "ethos", "items", "data", "objects"]:
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        nested = [v for v in payload.values() if isinstance(v, list)]
        if len(nested) == 1:
            return [x for x in nested[0] if isinstance(x, dict)]
    return []


def find_json_by_patterns(directory: Path, patterns: List[str]) -> Optional[Path]:
    files = sorted(directory.glob("*.json"))
    for f in files:
        name = f.name.lower()
        if all(any(p in name for p in patterns_part.split("|")) for patterns_part in patterns):
            return f
    for f in files:
        name = f.name.lower()
        if any(p in name for p in patterns):
            return f
    return None


def autodiscover_data(data_dir: Path, ethos_path: Optional[Path]) -> Tuple[Dict[str, Path], Path]:
    category_files: Dict[str, Path] = {}
    for category, pats in CATEGORY_PATTERNS.items():
        path = find_json_by_patterns(data_dir, pats)
        if not path:
            raise FileNotFoundError(f"No encontré JSON para categoría '{category}' en {data_dir}")
        category_files[category] = path

    if ethos_path:
        if not ethos_path.exists():
            raise FileNotFoundError(f"ETHOS file no existe: {ethos_path}")
        chosen_ethos = ethos_path
    else:
        candidate = find_json_by_patterns(data_dir, ["ethos", "persona"])
        if not candidate:
            raise FileNotFoundError("No encontré JSON de ETHOS. Usa --ethos-path.")
        chosen_ethos = candidate
    return category_files, chosen_ethos


def build_object_doc(item: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key in DOC_FIELDS:
        if key in item and item[key] is not None:
            val = " | ".join(flatten(item[key]))
            if val.strip():
                lines.append(f"{key}: {val}")
    if not lines:
        # Parser tolerante: si no hay campos estándar, usa todo el objeto.
        lines = flatten(item)
    return "\n".join(lines)


def ethos_doc(item: Dict[str, Any]) -> str:
    fields = [
        "name",
        "description",
        "principles",
        "commandments",
    ]
    parts: List[str] = []
    for f in fields:
        if f in item:
            val = " | ".join(flatten(item[f]))
            if val.strip():
                parts.append(f"{f}: {val}")
    return "\n".join(parts) if parts else " ".join(flatten(item))


def extract_output_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    try:
        pieces: List[str] = []
        for out in response.output:
            for c in out.content:
                if getattr(c, "type", "") in ("output_text", "text"):
                    if getattr(c, "text", None):
                        pieces.append(c.text)
        joined = "\n".join(pieces).strip()
        if joined:
            return joined
    except Exception:
        pass
    return str(response)


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No pude parsear JSON desde respuesta LLM.")


def replace_case_insensitive(text: str, target: str, replacement: str) -> str:
    if not target:
        return text
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    return pattern.sub(replacement, text)


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, model: str, key: str) -> Optional[List[float]]:
        return self.data.get(model, {}).get(key)

    def set(self, model: str, key: str, vec: List[float]) -> None:
        self.data.setdefault(model, {})[key] = vec

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")


class NaninRunner:
    def __init__(
        self,
        data_dir: Path,
        ethos_path: Optional[Path],
        model: str,
        embedding_model: str,
        seed: Optional[int],
        use_embeddings: bool,
        cache_path: Path,
        include_why: bool,
        api_key: Optional[str] = None,
        fast_mode: bool = False,
        include_metadata: bool = True,
    ):
        self.data_dir = data_dir
        self.model = model
        self.embedding_model = embedding_model
        self.seed = seed
        self.use_embeddings = use_embeddings
        self.include_why = include_why
        self.fast_mode = fast_mode
        self.include_metadata = include_metadata
        self.rand = random.Random(seed)
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.cache = EmbeddingCache(cache_path)

        category_files, ethos_file = autodiscover_data(data_dir, ethos_path)
        self.categories = {
            name: normalize_objects(read_json_file(path))
            for name, path in category_files.items()
        }
        self.ethos = normalize_objects(read_json_file(ethos_file))
        if not self.ethos:
            raise ValueError("ETHOS JSON cargado pero vacío o inválido.")
        self.usage_counts: Dict[str, int] = {}

    def ethos_capabilities(self, ethos: Dict[str, Any]) -> str:
        desc = " ".join(flatten(ethos.get("description", "")))
        principles = flatten(ethos.get("principles", []))[:3]
        cmd = flatten(ethos.get("commandments", []))[:2]
        chunks = []
        if desc:
            chunks.append(desc)
        if principles:
            chunks.append("Principles: " + " | ".join(principles))
        if cmd:
            chunks.append("Execution style: " + " | ".join(cmd))
        return " ".join(chunks).strip()

    def object_capabilities(self, item: Dict[str, Any]) -> str:
        fields = [
            "council_objective",
            "logic_narrative",
            "framework_lens",
            "when_to_use",
            "key_questions",
            "acceptance_criteria",
            "expected_outputs",
        ]
        out: List[str] = []
        for f in fields:
            if f in item and item[f] is not None:
                val = " | ".join(flatten(item[f])).strip()
                if val:
                    out.append(f"{f}: {val}")
        return "\n".join(out).strip()

    def sanitize_main_prompt(self, text: str, selected_item: Dict[str, Any], ethos: Dict[str, Any]) -> str:
        cleaned = text or ""
        banned_terms = [
            str(selected_item.get("object", "")).strip(),
            str(selected_item.get("slug_id", "")).strip(),
            str(ethos.get("name", "")).strip(),
            str(ethos.get("slug_id", "")).strip(),
        ]
        for term in banned_terms:
            if term:
                cleaned = replace_case_insensitive(cleaned, term, "this operating stance")

        # Evita nomenclatura interna tipo "BEAST – X", "ARTIFACT - Y", etc.
        cleaned = re.sub(
            r"\b(ARTIFACT|BEAST|BOND|CARD|GLAMOUR|RITUAL|WEAPON|ETHOS)\s*[–\-—]\s*[^\n\r]+",
            "domain-specific method",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Elimina placeholders sin resolver que rompen ejecución en LLM siguiente.
        cleaned = cleaned.replace("{geometry}", "triangle/funnel/flywheel/loop (choose one)")
        cleaned = cleaned.replace("{terms}", "2-4")
        cleaned = re.sub(r"\{[^{}]{1,40}\}", "defined parameter", cleaned)
        return cleaned.strip()

    def embedding_key(self, item_id: str, doc: str) -> str:
        h = hashlib.sha256(doc.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"{item_id}:{h}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in response.data]

    def get_or_create_embeddings(self, items: List[Dict[str, Any]]) -> List[List[float]]:
        docs = [build_object_doc(i) for i in items]
        ids = [
            i.get("slug_id")
            or safe_slug(i.get("object", "unknown"))
            or f"item-{idx}"
            for idx, i in enumerate(items)
        ]
        keys = [self.embedding_key(ids[idx], docs[idx]) for idx in range(len(items))]
        vectors: List[Optional[List[float]]] = [self.cache.get(self.embedding_model, k) for k in keys]

        missing_positions = [idx for idx, v in enumerate(vectors) if v is None]
        if missing_positions:
            missing_texts = [docs[idx] for idx in missing_positions]
            new_vectors = self.embed_texts(missing_texts)
            for pos, vec in zip(missing_positions, new_vectors):
                vectors[pos] = vec
                self.cache.set(self.embedding_model, keys[pos], vec)
            self.cache.save()
        return [v if v is not None else [] for v in vectors]

    def top_k_candidates(
        self,
        user_input: str,
        items: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        if self.use_embeddings:
            item_vecs = self.get_or_create_embeddings(items)
            query_vec = self.embed_texts([user_input])[0]
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for item, vec in zip(items, item_vecs):
                score = cosine(query_vec, vec)
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [{"score": s, "item": it} for s, it in scored[:k]]

        # Fallback sin embeddings: ranking por LLM sobre muestra acotada.
        sample = items[: min(20, len(items))]
        payload = []
        for idx, it in enumerate(sample):
            payload.append(
                {
                    "index": idx,
                    "slug_id": it.get("slug_id"),
                    "object": it.get("object"),
                    "logic_narrative": it.get("logic_narrative"),
                    "when_to_use": it.get("when_to_use"),
                }
            )
        prompt = (
            "Rankea relevancia semántica del input para estos objetos. "
            "Devuelve JSON exacto: {\"ranked_indices\":[...]}.\n"
            f"INPUT:\n{user_input}\n\nOBJETOS:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        rsp = self.client.responses.create(model=self.model, input=prompt)
        ranked = parse_json_from_text(extract_output_text(rsp)).get("ranked_indices", [])
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked[:k]):
            if isinstance(idx, int) and 0 <= idx < len(sample):
                out.append({"score": float(len(sample) - rank), "item": sample[idx]})
        if not out:
            out = [{"score": 0.0, "item": it} for it in sample[:k]]
        return out

    def choose_ethos(self, user_input: str) -> Dict[str, Any]:
        candidates = self.top_k_candidates(user_input, self.ethos, k=3)
        # 60/30/10 sobre top-3
        weights = [0.6, 0.3, 0.1][: len(candidates)]
        picked = self.rand.choices(candidates, weights=weights, k=1)[0]["item"]
        return picked

    def relevance_floor(self, category: str, user_input: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        candidate_payload = []
        for idx, c in enumerate(candidates):
            it = c["item"]
            candidate_payload.append(
                {
                    "index": idx,
                    "slug_id": it.get("slug_id"),
                    "object": it.get("object"),
                    "logic_narrative": it.get("logic_narrative"),
                    "when_to_use": it.get("when_to_use"),
                    "score": c.get("score", 0),
                }
            )
        prompt = (
            "Evalúa candidatos y aplica relevance floor.\n"
            "Regla: pasa solo si puedes justificar en 1 frase el vínculo input -> logic_narrative.\n"
            "Devuelve JSON exacto: {\"pass_indices\":[int],\"justifications\":{\"0\":\"...\"}}.\n"
            f"CATEGORIA: {category}\nINPUT: {user_input}\nCANDIDATOS: {json.dumps(candidate_payload, ensure_ascii=False)}"
        )
        rsp = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "low" if self.fast_mode else "medium"},
            input=prompt,
        )
        parsed = parse_json_from_text(extract_output_text(rsp))
        pass_indices = parsed.get("pass_indices", [])
        justifications = parsed.get("justifications", {})
        passed: List[Dict[str, Any]] = []
        for idx in pass_indices:
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                entry = dict(candidates[idx])
                entry["why_selected"] = justifications.get(str(idx)) or justifications.get(idx) or ""
                passed.append(entry)
        return passed or candidates[:2]

    def stochastic_pick(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Peso base 70/20/10 distribuido por rangos.
        n = len(candidates)
        if n == 1:
            return candidates[0]
        base_weights = []
        for idx in range(n):
            if idx == 0:
                w = 0.70
            elif idx in (1, 2):
                w = 0.20 / min(2, max(1, n - 1))
            else:
                tail = n - 3
                w = (0.10 / tail) if tail > 0 else 0.01
            # Penalización de novedad (uso reciente/acumulado)
            item = candidates[idx]["item"]
            slug = item.get("slug_id") or safe_slug(item.get("object", f"idx-{idx}"))
            penalty = 1.0 / (1.0 + self.usage_counts.get(slug, 0))
            base_weights.append(max(1e-6, w * penalty))
        chosen = self.rand.choices(candidates, weights=base_weights, k=1)[0]
        slug = chosen["item"].get("slug_id") or safe_slug(chosen["item"].get("object", "unknown"))
        self.usage_counts[slug] = self.usage_counts.get(slug, 0) + 1
        return chosen

    def get_reasoning_style_pool(self) -> List[str]:
        # For output portability, use plain-language lenses, not internal card names.
        return [
            "Systems Mapping",
            "Trade-off Analysis",
            "Adversarial Framing",
            "Second-order Effects",
            "Constraint-first Design",
            "Failure-mode Backcasting",
            "Scenario Stress Testing",
            "Root-cause Decomposition",
            "Decision-focused Prioritization",
            "Signal-vs-Noise Filtering",
            "Hypothesis-driven Evaluation",
            "Risk-weighted Planning",
            "Counterfactual Comparison",
            "Temporal Horizon Planning",
            "Execution Sequencing",
        ]

    def generate_main_prompt(
        self,
        category: str,
        selected_item: Dict[str, Any],
        ethos: Dict[str, Any],
        user_input: str,
    ) -> Tuple[str, str]:
        styles = self.get_reasoning_style_pool()
        ritual_guardrail = (
            "RITUAL RULE: If category is Rituals, use the ritual as-is from fields provided. "
            "Do not invent or reinterpret a new ritual concept."
        )
        ethos_caps = self.ethos_capabilities(ethos)
        object_caps = self.object_capabilities(selected_item)
        prompt = (
            "Genera UN Main Prompt listo para copiar/pegar.\n"
            "Debe estar anclado al logic_narrative y NO en chat libre.\n"
            "Incluye exactamente estas secciones en este orden:\n"
            "1) Role\n2) Reasoning Styles (1-3)\n3) Context Summary\n4) Objective + Deliverable Format\n"
            "5) Constraints & Exclusions\n6) Definition of Success (3-6 checks)\n"
            "Regla dura: NO menciones nombres internos, ids, slugs, marcas registradas, ni etiquetas como "
            "Selector / ARTIFACT / BEAST / CARD / GLAMOUR / RITUAL / WEAPON / ETHOS.\n"
            "Describe capacidades en lenguaje natural portable para cualquier LLM.\n"
            f"{ritual_guardrail}\n"
            "Devuelve JSON exacto: {\"main_prompt\":\"...\",\"why_selected\":\"...\"}.\n"
            f"USER_INPUT: {user_input}\n"
            f"CATEGORY: {category}\n"
            f"ETHOS_CAPABILITIES: {ethos_caps}\n"
            f"REASONING_STYLE_POOL: {json.dumps(styles, ensure_ascii=False)}\n"
            f"SELECTED_OBJECT_CAPABILITIES: {object_caps}\n"
        )
        rsp = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=prompt,
        )
        parsed = parse_json_from_text(extract_output_text(rsp))
        main_prompt = self.sanitize_main_prompt(parsed.get("main_prompt", "").strip(), selected_item, ethos)
        return main_prompt, parsed.get("why_selected", "").strip()

    def run(
        self,
        user_input: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        selected_categories: Optional[List[str]] = None,
        selected_artifact_slugs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        def emit(event: Dict[str, Any]) -> None:
            if progress_callback:
                progress_callback(event)

        categories_to_run = [c for c in (selected_categories or DEFAULT_CATEGORY_ORDER) if c in self.categories]
        if not categories_to_run:
            raise ValueError("No selected categories to run.")

        emit({"stage": "start"})
        ethos = self.choose_ethos(user_input)
        emit({"stage": "ethos_selected", "ethos_name": ethos.get("name", "Unknown ETHOS")})
        selections: Dict[str, Dict[str, Any]] = {}

        for category in categories_to_run:
            items = list(self.categories.get(category, []))
            if category == "Artifacts" and selected_artifact_slugs:
                allowed = {s.strip() for s in selected_artifact_slugs if s and s.strip()}
                items = [it for it in items if str(it.get("slug_id", "")).strip() in allowed]
                if not items:
                    raise ValueError("No valid artifacts match your selected artifact slugs.")
            k = 3 if self.fast_mode else 5
            top5 = self.top_k_candidates(user_input, items, k=k)
            filtered = top5[:2] if self.fast_mode else self.relevance_floor(category, user_input, top5)
            chosen = self.stochastic_pick(filtered)
            selections[category] = chosen
            emit({"stage": "selection_done", "category": category})

        output_items = []
        for category in categories_to_run:
            chosen = selections[category]
            item = chosen["item"]
            main_prompt, why = self.generate_main_prompt(category, item, ethos, user_input)
            output_item = {
                "category": category,
                "selected_object": {
                    "slug_id": item.get("slug_id") or safe_slug(item.get("object", category)),
                    "name": item.get("object") or item.get("name") or "Unnamed",
                },
                "main_prompt": main_prompt,
            }
            if self.include_why:
                output_item["why_selected"] = why or chosen.get("why_selected", "")
            output_items.append(output_item)
            emit({"stage": "prompt_done", "category": category, "item": output_item})
        if self.include_metadata:
            result = {
                "ethos_selected": {
                    "id": ethos.get("slug_id") or safe_slug(ethos.get("name", "ethos")),
                    "name": ethos.get("name", "Unknown ETHOS"),
                },
                "seed": self.seed,
                "items": output_items,
            }
        else:
            result = {
                "items": [{"category": i["category"], "main_prompt": i["main_prompt"]} for i in output_items],
            }
        emit({"stage": "completed"})
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NANIN 6-Prompt Runner (local)")
    parser.add_argument("--input", required=True, help="Input libre del usuario")
    parser.add_argument("--data-dir", default=".", help="Carpeta con JSON de categorías")
    parser.add_argument("--ethos-path", default=None, help="Ruta JSON ETHOS opcional")
    parser.add_argument("--model", default="gpt-5.2-codex", help="Modelo Responses para selección/generación")
    parser.add_argument("--embedding-model", default="text-embedding-3-large", help="Modelo de embeddings")
    parser.add_argument("--seed", type=int, default=None, help="Seed opcional para reproducibilidad parcial")
    parser.add_argument("--no-embeddings", action="store_true", help="Desactiva embeddings y usa ranking LLM")
    parser.add_argument("--cache-path", default=".nanin_cache/embeddings.json", help="Cache local de embeddings")
    parser.add_argument("--no-why", action="store_true", help="Oculta why_selected")
    parser.add_argument("--api-key", default=None, help="API key opcional (si no, usa OPENAI_API_KEY)")
    parser.add_argument("--fast", action="store_true", help="Modo rápido: menos llamadas y menor latencia")
    parser.add_argument("--clean-output", action="store_true", help="Oculta metadata interna y devuelve solo category+main_prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = NaninRunner(
        data_dir=Path(args.data_dir),
        ethos_path=Path(args.ethos_path) if args.ethos_path else None,
        model=args.model,
        embedding_model=args.embedding_model,
        seed=args.seed,
        use_embeddings=not args.no_embeddings,
        cache_path=Path(args.cache_path),
        include_why=not args.no_why,
        api_key=args.api_key,
        fast_mode=args.fast,
        include_metadata=not args.clean_output,
    )
    result = runner.run(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
