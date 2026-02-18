import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from nanin_runner import DEFAULT_CATEGORY_ORDER, NaninRunner


st.set_page_config(page_title="NANIN Prompt Runner", page_icon="🧭", layout="wide")
BASE_DIR = Path(__file__).resolve().parent


def load_artifacts() -> List[Dict[str, str]]:
    path = BASE_DIR / "Nanin Artifacts.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, list):
            return []
        out = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug_id", "")).strip()
            if not slug:
                continue
            name = str(item.get("object", slug)).strip()
            when = item.get("when_to_use")
            brief = ""
            if isinstance(when, list) and when:
                brief = str(when[0])
            elif isinstance(item.get("council_objective"), str):
                brief = str(item.get("council_objective"))
            out.append({"slug": slug, "name": name, "brief": brief})
        return out
    except Exception:
        return []


def get_api_key(user_key: str) -> Optional[str]:
    user_key = (user_key or "").strip()
    if user_key:
        return user_key
    return st.secrets.get("OPENAI_API_KEY")


def render_category_help() -> None:
    st.caption("What each category is best for:")
    st.markdown("- `Artifacts`: leverage framing and analytical structure in ambiguous situations.")
    st.markdown("- `Bonds & Beasts`: stakeholder dynamics, trust/fear patterns, behavior under pressure.")
    st.markdown("- `Cards`: tactical reasoning lenses for concrete decision moves.")
    st.markdown("- `Glamour`: narrative packaging and persuasion salience.")
    st.markdown("- `Rituals`: repeatable routines as-defined in the source cards.")
    st.markdown("- `Weapons`: decisive interventions and adversarial stress tests.")


def main() -> None:
    st.title("NANIN Prompt Runner")
    st.write("Generate only the categories you need, with optional Artifact-level control.")

    with st.sidebar:
        st.subheader("Run Settings")
        api_key_input = st.text_input("OpenAI API key (optional)", type="password", placeholder="sk-...")
        model = st.text_input("Model", value="gpt-5.2-codex")
        seed_text = st.text_input("Seed (optional)", value="")
        fast_mode = st.checkbox("Fast mode", value=True)
        no_embeddings = st.checkbox("No embeddings", value=False)
        include_why = st.checkbox("Include why_selected", value=False)
        show_metadata = st.checkbox("Show internal metadata", value=False)

    user_input = st.text_area("Input", height=140, placeholder="Describe your business/personal/testing case...")

    render_category_help()
    selected_categories = st.multiselect(
        "Categories to generate",
        options=DEFAULT_CATEGORY_ORDER,
        default=DEFAULT_CATEGORY_ORDER,
    )

    artifacts = load_artifacts()
    artifact_options = {f"{a['name']}": a["slug"] for a in artifacts}
    with st.expander("Artifacts filter (optional)", expanded=False):
        st.caption("If `Artifacts` is selected, choose specific artifacts to force candidate selection.")
        selected_artifact_labels = st.multiselect(
            "Artifacts",
            options=list(artifact_options.keys()),
            default=[],
        )
        for art in artifacts:
            st.markdown(f"- `{art['name']}`: {art['brief']}")

    if st.button("Generate", type="primary", use_container_width=True):
        api_key = get_api_key(api_key_input)
        if not api_key:
            st.error("Set API key in sidebar or `.streamlit/secrets.toml` as OPENAI_API_KEY.")
            st.stop()
        if not user_input.strip():
            st.error("Input is required.")
            st.stop()
        if not selected_categories:
            st.error("Select at least one category.")
            st.stop()

        seed: Optional[int] = None
        if seed_text.strip():
            try:
                seed = int(seed_text.strip())
            except ValueError:
                st.error("Seed must be an integer.")
                st.stop()

        selected_artifact_slugs = [artifact_options[label] for label in selected_artifact_labels]
        status = st.status("Running pipeline...", expanded=True)
        progress = st.progress(0)
        result_box = st.container()
        partial_items: List[Dict[str, str]] = []

        try:
            runner = NaninRunner(
                data_dir=BASE_DIR,
                ethos_path=None,
                model=model,
                embedding_model="text-embedding-3-large",
                seed=seed,
                use_embeddings=not no_embeddings,
                cache_path=BASE_DIR / ".nanin_cache" / "embeddings.json",
                include_why=include_why,
                api_key=api_key,
                fast_mode=fast_mode,
                include_metadata=show_metadata,
            )

            total = max(1, len(selected_categories))

            def on_progress(event: Dict[str, str]) -> None:
                stage = event.get("stage", "")
                if stage == "ethos_selected":
                    status.write(f"ETHOS selected: {event.get('ethos_name', 'unknown')}")
                elif stage == "selection_done":
                    status.write(f"Selection done: {event.get('category')}")
                elif stage == "prompt_done":
                    item = event.get("item")
                    if isinstance(item, dict):
                        partial_items.append(item)
                    progress.progress(min(len(partial_items) / total, 1.0))
                    with result_box:
                        st.subheader("Generated Prompts")
                        for i in partial_items:
                            st.markdown(f"### {i.get('category','Category')}")
                            st.code(i.get("main_prompt", ""), language="markdown")

            result = runner.run(
                user_input=user_input.strip(),
                progress_callback=on_progress,
                selected_categories=selected_categories,
                selected_artifact_slugs=selected_artifact_slugs,
            )

            progress.progress(1.0)
            status.update(label="Completed", state="complete", expanded=False)

            with result_box:
                st.subheader("Final Output")
                for item in result.get("items", []):
                    st.markdown(f"### {item.get('category','Category')}")
                    st.code(item.get("main_prompt", ""), language="markdown")
                if show_metadata:
                    st.json(result)
        except Exception as exc:
            status.update(label="Error", state="error", expanded=True)
            st.error(str(exc))


if __name__ == "__main__":
    main()
