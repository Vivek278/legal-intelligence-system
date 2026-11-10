"""Split the original experiments using configuration selections only.

The source notebooks remain available in Git history at SOURCE_COMMIT. Run
`python scripts/reorganize_notebooks.py --write` to reproduce the split, or
`python scripts/reorganize_notebooks.py --check` to audit the working tree.
Requires nbformat and IPython for validation; generation uses the standard library.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "7e7c8a9f1605c5f9836a3cb2cb7c28004343f5f3"
SOURCES = {
    "p2": ("phase2/capstone-p2-4methods.ipynb", 4),
    "p2_extra": ("phase2/phase2-6methods.ipynb", 3),
    "p3": ("phase3/capstone-p3-model-4-and-6.ipynb", 3),
    "p3_m7": ("phase3/capstone-p3-model7.ipynb", 3),
    "p4": ("phase4/capstone-p4-all-models-all-methods.ipynb", 4),
    "p5": ("phase5/capstone-p5-all-models-all-methods.ipynb", 5),
    "p6": ("phase6/capstone-p6-all-models-all-methods.ipynb", 5),
    "p7": ("phase7/capstone-p7-all-models-all-methods.ipynb", 5),
}
TASKS = {
    2: "Legal Named Entity Recognition",
    3: "Rhetorical Role Prediction",
    4: "Court Judgment Prediction",
    5: "Legal Statute Identification",
    6: "Abstractive Legal Summarization",
    7: "Prior Case Retrieval",
}
LABELS = {
    "full": "Full fine-tuning", "partial": "Partial fine-tuning",
    "lora": "LoRA", "lorma": "LoRMA", "kd": "Knowledge distillation",
    "houlsby": "Houlsby adapters", "pfeiffer": "Pfeiffer adapters",
    "prefix": "Prefix tuning", "prompt": "Prompt tuning", "bitfit": "BitFit",
    "ensemble": "Ensemble",
}
EXTRA_METHODS = ["houlsby", "pfeiffer", "prefix", "prompt", "bitfit", "kd"]


def git_bytes(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def load_sources():
    return {
        key: json.loads(git_bytes("show", f"{SOURCE_COMMIT}:{path}"))
        for key, (path, _) in SOURCES.items()
    }


def replace_assignments(source, selections):
    """Replace only RHS text of named top-level assignments; retain all other text."""
    lines = source.splitlines(keepends=True)
    changes = []
    found = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names = ([target.id] if isinstance(target, ast.Name) else
                     [e.id for e in target.elts] if isinstance(target, ast.Tuple) else [])
            key = ",".join(names)
            if key not in selections or key in found:
                continue
            value = node.value
            # AST column offsets are UTF-8 bytes, not Unicode character positions.
            start = sum(map(len, lines[:value.lineno - 1])) + len(
                lines[value.lineno - 1].encode("utf-8")[:value.col_offset].decode("utf-8"))
            end = sum(map(len, lines[:value.end_lineno - 1])) + len(
                lines[value.end_lineno - 1].encode("utf-8")[:value.end_col_offset].decode("utf-8"))
            changes.append((start, end, selections[key]))
            found.add(key)
    if found != set(selections):
        raise ValueError(f"Missing config assignments: {set(selections) - found}")
    for start, end, value in sorted(changes, reverse=True):
        source = source[:start] + value + source[end:]
    return source


def experiments():
    specs = []

    def add(phase, model, method):
        if phase == 2:
            key = "p2_extra" if method in EXTRA_METHODS or method == "ensemble" else "p2"
            # LoRA/LoRMA are not in EXTRA_METHODS. Additional methods are M6-only.
            if key == "p2_extra":
                assert model == 6
                config = {"METHODS_TO_RUN": repr(EXTRA_METHODS if method == "ensemble" else [method]),
                          "RUN_ENSEMBLE": repr(method == "ensemble")}
            else:
                config = {"METHOD": repr(method)}
                if model == 4:
                    # These exact M4 settings are present as comments in the source config.
                    config.update({
                        "D_MODEL,N_LAYERS,N_HEADS,N_KV_HEADS": "768, 16, 12, 2",
                        "HEAD_DIM,D_FF,MAX_SEQ_LEN": "D_MODEL // N_HEADS, 2048, 1024",
                        "ROPE_BASE,WINDOW_SIZE": "10000.0, 256",
                    })
        elif phase == 3:
            key = "p3_m7" if model == 7 else "p3"
            config = {"FUSE_METHOD": repr(method)}
            if model != 7:
                config["PH1_OPTION"] = repr(1 if model == 4 else 2)
        else:
            key = f"p{phase}"
            chosen = (["full", "lora", "lorma", "knowledge_distillation"] if method == "ensemble"
                      else ["knowledge_distillation" if phase in (6, 7) and method == "kd" else method])
            config = {"RUN_METHODS": repr(chosen)}
            config["ARCH" if phase == 5 else "MODEL_OPTION"] = repr(f"model{model}" if phase == 5 else model)
        path = f"phase{phase}/model{model}/{method}.ipynb"
        # Separate output directories are IO-only: all model, loss, metric and
        # training function bodies remain verbatim from the original notebook.
        run_id = method
        if phase == 2:
            config["OUT_ROOT" if key == "p2_extra" else "OUT_DIR"] = repr(
                f"/kaggle/working/phase2_model{model}_{run_id}")
        elif phase == 3:
            config["OUT_DIR"] = repr(f"/kaggle/working/phase3_model{model}_{run_id}")
        elif phase == 4:
            config["OUTPUT_DIR"] = f"OUTPUT_ROOT / 'cjpe_p4_model{model}_{run_id}'"
        else:
            stem = {5: "lsi_p5", 6: "summ_p6", 7: "pcr_p7"}[phase]
            fallback = "." if phase == 6 else "/mnt/data"
            name = f"{stem}_model{model}_{run_id}"
            config["OUTPUT_DIR"] = (f"Path('/kaggle/working/{name}' if Path('/kaggle/working').exists() "
                                    f"else '{fallback}/{name}')")
        specs.append(dict(phase=phase, model=model, method=method,
                          source_key=key, path=path, selections=config))

    for model in (4, 6):
        for method in ("full", "partial", "lora", "lorma"):
            add(2, model, method)
    for method in EXTRA_METHODS + ["ensemble"]:
        add(2, 6, method)
    for phase in range(3, 8):
        for model in (4, 6, 7):
            for method in ("full", "lora", "lorma", "kd"):
                add(phase, model, method)
            if phase == 7:
                add(phase, model, "ensemble")

    return specs


def render_notebook(spec, sources):
    key = spec["source_key"]
    original = sources[key]
    notebook = copy.deepcopy(original)
    source_path, config_cell = SOURCES[key]
    config_source = "".join(notebook["cells"][config_cell]["source"])
    notebook["cells"][config_cell]["source"] = replace_assignments(
        config_source, spec["selections"]).splitlines(keepends=True)
    phase, model, method = spec["phase"], spec["model"], spec["method"]
    title = f"Phase {phase} — {TASKS[phase]} — Model {model} — {LABELS[method]}"
    note = (
        f"# {title}\n\n"
        "Run this notebook from top to bottom in a fresh kernel with the matching "
        "foundation checkpoint and tokenizer. The experiment is already selected in the configuration cell.\n\n"
        f"**Source:** `{source_path}` at commit `{SOURCE_COMMIT}`. "
        "Model/training/evaluation/submission implementations are copied unchanged; "
        "only configuration selections and output directories differ. Saved outputs "
        "and execution counts were cleared during organization.\n\n"
        "**Outputs:** use a separate Kaggle session or working directory for each run. "
        "The original submission filenames are preserved, so some ZIP files at the "
        "working-directory root can be overwritten by another run.\n\n"
    )
    if method == "ensemble":
        note += ("**Ensemble dependencies:** this workflow intentionally runs its constituent "
                 "methods in this kernel before applying the original in-memory ensemble. "
                 "It does not load results from the individual notebooks.\n\n")
    elif method == "kd":
        note += ("**Distillation:** the teacher/student workflow remains intact. "
                 "The original code trains or reuses a full teacher before training the student.\n\n")
    note += "## Original implementation notes\n\n" + "".join(original["cells"][0]["source"])
    notebook["cells"][0]["source"] = note.splitlines(keepends=True)
    for i, cell in enumerate(notebook["cells"]):
        cell["metadata"].pop("execution", None)
        if notebook["nbformat_minor"] >= 5:
            cell["id"] = hashlib.sha256(f"{spec['path']}:{i}".encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    # Kernel and language metadata are useful; source Kaggle execution IDs are not.
    notebook["metadata"] = {k: copy.deepcopy(v) for k, v in original["metadata"].items()
                            if k in ("kernelspec", "language_info")}
    notebook["metadata"]["experiment"] = {
        "source_commit": SOURCE_COMMIT, "source_notebook": source_path,
        "source_config_cell": config_cell, "model": model, "method": method,
        "selections": spec["selections"],
    }
    return notebook


def phase_readme(phase, specs):
    entries = [s for s in specs if s["phase"] == phase]
    lines = [f"# Phase {phase}: {TASKS[phase]}", "",
             "Open a model folder, then choose an adaptation notebook. Each notebook "
             "contains the original setup, data loading, model, training, evaluation, and "
             "submission code, with that experiment preselected. Run top to bottom in a fresh kernel.", "",
             "| Model | Adaptation notebooks |", "|---|---|"]
    for model in sorted({s["model"] for s in entries}):
        methods = [s for s in entries if s["model"] == model]
        links = ", ".join(f"[{LABELS[s['method']]}]({Path(s['path']).relative_to(f'phase{phase}').as_posix()})" for s in methods)
        lines.append(f"| Model {model} | {links} |")
    if phase == 2:
        lines += ["", "Models 4 and 6 use the four core methods. The six additional methods "
                  "and their logit ensemble use the existing Model 6 configuration. In particular, "
                  "the extra-method notebook's `KD_KEEP` map is specific to its 20-layer backbone. "
                  "No Model 4 distillation map or Model 7 NER implementation has been invented."]
    if phase in (2, 7):
        lines += ["", "The ensemble notebook keeps the original in-memory dependencies: it runs "
                  "the constituent methods before averaging their predictions. Single-method "
                  "notebooks run only their selected method (plus a teacher when needed for KD)."]
    lines += ["", "Representation, pooling, loss, thresholding, and other existing switches "
              "remain configurable inside the notebooks. Their original defaults are retained."]
    lines += ["", "## Running and provenance", "",
              "See the root [README](../README.md) for prerequisites, checkpoint/tokenizer setup, "
              "dataset access, and results from the report. Each notebook's first cell and "
              "`metadata.experiment` record its source and configuration selections.", "",
              "Experiment output folders are separate, but original root-level submission names "
              "are preserved. Use separate sessions/working directories or download submissions "
              "after each run. Saved outputs were cleared; no training was rerun for this split.", ""]
    return "\n".join(lines)


def validate(specs, sources):
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    transformer = TransformerManager()
    expected_paths = {s["path"] for s in specs}
    actual_paths = {p.relative_to(ROOT).as_posix() for phase in range(2, 8)
                    for p in (ROOT / f"phase{phase}").rglob("*.ipynb")}
    assert actual_paths == expected_paths, ("Notebook inventory differs", actual_paths ^ expected_paths)
    code_cells = 0
    unchanged_cells = 0
    for spec in specs:
        actual = json.loads((ROOT / spec["path"]).read_text(encoding="utf-8"))
        expected = render_notebook(spec, sources)
        assert actual == expected, f"Notebook differs from source/config selections: {spec['path']}"
        nbformat.validate(nbformat.from_dict(actual))
        _, config_cell = SOURCES[spec["source_key"]]
        original = sources[spec["source_key"]]
        for i, cell in enumerate(actual["cells"]):
            if cell["cell_type"] != "code":
                continue
            code_cells += 1
            source = "".join(cell["source"])
            if i != config_cell:
                assert source == "".join(original["cells"][i]["source"])
                unchanged_cells += 1
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile(transformer.transform_cell(source), f"{spec['path']}:cell{i}", "exec")
    # Byte-for-byte audit also catches untracked changes to protected Phase 1 files.
    phase1_files = git_bytes("ls-tree", "-r", "--name-only", SOURCE_COMMIT, "phase1").decode().splitlines()
    assert sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "phase1").rglob("*") if p.is_file()) == sorted(phase1_files)
    for path in phase1_files + ["CS781-CP-Vivek-Chandwani-241183-report.pdf"]:
        assert (ROOT / path).read_bytes() == git_bytes("show", f"{SOURCE_COMMIT}:{path}"), f"Protected file changed: {path}"
    for phase in range(2, 8):
        assert (ROOT / f"phase{phase}/README.md").read_text(encoding="utf-8") == phase_readme(phase, specs)
    print(f"Validated {len(specs)} notebooks / {code_cells} code cells; "
          f"{unchanged_cells} non-config code cells are verbatim. "
          "Notebook schema and Python/IPython syntax pass. Phase 1 and the report are byte-identical.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sources = load_sources()
    specs = experiments()
    if args.write:
        for spec in specs:
            path = ROOT / spec["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(render_notebook(spec, sources), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for phase in range(2, 8):
            (ROOT / f"phase{phase}/README.md").write_text(phase_readme(phase, specs), encoding="utf-8")
        # Remove only the eight explicitly listed originals after their replacements exist.
        for source_path, _ in SOURCES.values():
            path = ROOT / source_path
            assert path.parent in {ROOT / f"phase{p}" for p in range(2, 8)}
            if path.exists():
                path.unlink()
        print(f"Wrote {len(specs)} notebooks and six phase indexes.")
    validate(specs, sources)


if __name__ == "__main__":
    main()
