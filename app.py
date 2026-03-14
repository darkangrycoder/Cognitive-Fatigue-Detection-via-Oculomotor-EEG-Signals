"""
Cognitive Fatigue Detector — HuggingFace Spaces App
====================================================
Compatible with Gradio 5.9.1.

THE FIX for 'TypeError: argument of type bool is not iterable':

  Root cause: Gradio's /info endpoint calls json_schema_to_python_type()
  on every component's JSON schema. gr.HTML produces a schema where
  additionalProperties is the Python boolean False. gradio_client then
  executes 'if "const" in False' which throws TypeError on every request.

  show_api=False does NOT fix this in 5.9.1 — it only hides the API UI
  but the /info route still executes and still crashes.

  Real fix: Replace ALL gr.HTML components with gr.Markdown.
  gr.Markdown accepts and renders HTML content identically, but its JSON
  schema does not produce the boolean additionalProperties value that
  triggers the crash. Zero visual change to the user.
"""

import io
import os
import traceback

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage

import inference
from feature_extractor import extract_features_from_csv, extract_features_from_image

matplotlib.use("Agg")

REPO_ID = "tdnathmlenthusiast/cognitive-fatigue-detector"


# ── Startup ──────────────────────────────────────────────────────────────────

def startup():
    print("Loading model checkpoint ...")
    try:
        inference.load_checkpoint()
        print("Model ready.")
    except Exception as e:
        print(f"WARNING: Could not load model — {e}")
        print("Make sure gaze_clean_checkpoint.pt is uploaded to the model repo.")

startup()


# ── Utilities ─────────────────────────────────────────────────────────────────

def fig_to_pil(fig) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf).copy()
    plt.close(fig)
    return img


def load_plot(filename: str):
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"plots/{filename}",
            repo_type="model",
        )
        return PILImage.open(path)
    except Exception:
        return None


def make_feature_chart(features: dict, pop_stats: dict) -> PILImage.Image:
    cols = list(features.keys())
    vals = [features[c] for c in cols]
    meds = [float(pop_stats.get(c, {}).get("median", 0)) for c in cols]

    ratios = [(v / m * 100) if m and m != 0 else 100 for v, m in zip(vals, meds)]
    colors = ["#27AE60" if 70 <= r <= 130 else "#F39C12" for r in ratios]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cols))
    ax.bar(x, ratios, color=colors, alpha=0.85, width=0.6)
    ax.axhline(100, color="#666", lw=1.2, ls="--", label="Population median (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of population median")
    ax.set_title(
        "Extracted Features vs Population Median\n"
        "Green = within normal range (70-130%)   Amber = outside range",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig_to_pil(fig)


# ── Tab 1: Webcam ─────────────────────────────────────────────────────────────

def analyze_webcam(image_array):
    if image_array is None:
        return "<p style='color:#888;padding:12px;'>Capture a photo first.</p>", None

    if not inference.is_loaded():
        return "<p style='color:#c0392b;padding:12px;'>Model not loaded.</p>", None

    try:
        pop_stats = inference.get_population_stats()
        features, detection_status = extract_features_from_image(image_array, pop_stats)
        score    = inference.predict(features)
        insights = inference.score_to_insights(score, features)

        disclaimer = (
            "Webcam mode — approximate analysis. "
            "Temporal features (blink rate, saccade velocity) are estimated from "
            "population medians because they require time-series signal. "
            "Upload a GazeBase CSV for full precision."
        )
        html  = inference.build_insights_html(
            insights,
            source_label=f"Detection: {detection_status}",
            disclaimer=disclaimer,
        )
        chart = make_feature_chart(features, pop_stats)
        return html, chart

    except Exception:
        err = traceback.format_exc()
        return f"<pre style='color:red;font-size:12px;'>{err}</pre>", None


# ── Tab 2: CSV Upload ─────────────────────────────────────────────────────────

def analyze_csv(file_obj):
    if file_obj is None:
        return "<p style='color:#888;padding:12px;'>Upload a CSV file first.</p>", None

    if not inference.is_loaded():
        return "<p style='color:#c0392b;padding:12px;'>Model not loaded.</p>", None

    try:
        df = pd.read_csv(file_obj.name, sep=",")
    except Exception as e:
        return f"<p style='color:red;padding:12px;'>Could not read CSV: {e}</p>", None

    required = {"x", "y", "val", "dP"}
    missing  = required - set(df.columns)
    if missing:
        return (
            f"<p style='color:red;padding:12px;'>"
            f"Missing columns: {missing}. "
            f"Expected GazeBase format: n, x, y, val, dP, lab</p>",
            None,
        )

    try:
        features = extract_features_from_csv(df)
    except Exception as e:
        return f"<p style='color:red;padding:12px;'>Feature extraction failed: {e}</p>", None

    if features is None:
        return (
            "<p style='color:#c0392b;padding:12px;'>"
            "Fewer than 50 valid samples. Check the CSV file.</p>",
            None,
        )

    try:
        score     = inference.predict(features)
        insights  = inference.score_to_insights(score, features)
        pop_stats = inference.get_population_stats()

        source_label = (
            f"File: {os.path.basename(file_obj.name)} "
            f"· {len(df):,} rows · exact pipeline"
        )
        html  = inference.build_insights_html(insights, source_label=source_label)
        chart = make_feature_chart(features, pop_stats)
        return html, chart

    except Exception:
        err = traceback.format_exc()
        return f"<pre style='color:red;font-size:12px;'>{err}</pre>", None


# ── Tab 3: Results ────────────────────────────────────────────────────────────

def build_results_html() -> str:
    loso = inference.get_loso_results()
    s1   = inference.get_stage1_metrics()

    if not loso:
        return "<p style='color:#888;padding:12px;'>LOSO results not in checkpoint.</p>"

    aucs     = [r["auc"]        for r in loso]
    f1s      = [r["f1_macro"]   for r in loso]
    recs     = [r["drowsy_rec"] for r in loso]
    aucs_no6 = [r["auc"]        for r in loso if r["subject"] != 6]

    def card(label, value, sub=""):
        sub_html = (
            '<div style="font-size:11px;color:#aaa;margin-top:3px;">' + sub + '</div>'
            if sub else ""
        )
        return (
            '<div style="background:#f8f9fa;border:1px solid #e0e0e0;'
            'border-radius:8px;padding:12px 16px;text-align:center;">'
            '<div style="font-size:11px;color:#888;text-transform:uppercase;'
            'letter-spacing:0.05em;margin-bottom:6px;">' + label + '</div>'
            '<div style="font-size:22px;font-weight:600;color:#333;">' + value + '</div>'
            + sub_html + '</div>'
        )

    cards = "".join([
        card("Mean AUC (all 12)",  f"{np.mean(aucs):.3f}",     f"± {np.std(aucs):.3f}"),
        card("AUC excl. S6",       f"{np.mean(aucs_no6):.3f}", f"± {np.std(aucs_no6):.3f}"),
        card("F1-macro",           f"{np.mean(f1s):.3f}",      f"± {np.std(f1s):.3f}"),
        card("Drowsy Recall",      f"{np.mean(recs):.3f}",     f"± {np.std(recs):.3f}"),
        card("GazeBase R²",        f"{s1.get('r2', 0):.3f}",   f"MAE {s1.get('mae', 0):.3f}"),
    ])

    return (
        '<div style="font-family:sans-serif;">'
        '<h3 style="font-size:15px;font-weight:500;color:#333;margin:0 0 12px;">'
        'LOSO Cross-Validation — 12 SEED-VIG Subjects</h3>'
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);'
        'gap:10px;margin-bottom:16px;">' + cards + '</div>'
        '<p style="font-size:12px;color:#888;margin:0;">'
        'Leave-One-Subject-Out validation. Subject 6 flagged as outlier '
        '(AUC=0.73, recall=0.32). Clean 16-feature model.</p>'
        '</div>'
    )


# ── About ─────────────────────────────────────────────────────────────────────

ABOUT_MD = """
## Cognitive Fatigue Detector

Two-stage cross-modal transfer learning for fatigue detection from eye-tracking signals.

### Pipeline

```
GazeBase (881 subjects, 1000 Hz)
  → 16 oculomotor features
  → SharedEncoder pretraining (fatigue regression)
  → 64-dim latent representation

SEED-VIG (12 subjects, 200 Hz EEG)
  → 204 spectral features
  → SharedEncoder fine-tuning (LOSO, alert vs drowsy)
  → AUC = 0.906 excl. outlier S6
```

### Features (16 clean biomarkers)

| Group | Features |
|---|---|
| Blink | blink_rate_pm, blink_count, mean_blink_ms |
| Fixation | num_fixations, mean_fix_ms, std_fix_ms |
| Saccade | num_saccades, mean_sac_ms, mean_sac_vel, peak_sac_vel, vel_std |
| Pupil | mean_pupil, std_pupil, pupil_range |
| Gaze dispersion | gaze_x_std, gaze_y_std |

### Why LOSO?

Leave-One-Subject-Out is the correct protocol for small-N EEG studies.
A random split would mix windows from the same person across train and test,
producing inflated metrics that don't reflect real-world generalisation.

### Webcam limitations

Single-frame capture cannot measure temporal features. Those fall back to
population medians from training. Upload a GazeBase CSV for the exact pipeline.

### References

- GazeBase: Griffith et al., 2021, *Scientific Data*, doi:10.1038/s41597-021-00959-y
- SEED-VIG: Zheng & Lu, 2018, *IEEE TNSRE*
- Model: [tdnathmlenthusiast/cognitive-fatigue-detector](https://huggingface.co/tdnathmlenthusiast/cognitive-fatigue-detector)
"""


# ── Gradio app ────────────────────────────────────────────────────────────────
# KEY: ALL gr.HTML replaced with gr.Markdown.
# gr.Markdown renders HTML content identically but its JSON schema does NOT
# produce additionalProperties=false (boolean), which is the value that causes
# gradio_client to throw TypeError: argument of type bool is not iterable.

with gr.Blocks(title="Cognitive Fatigue Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Cognitive Fatigue Detector\n"
        "Two-stage transfer learning · GazeBase pretraining → SEED-VIG LOSO · "
        "[Model on HuggingFace](https://huggingface.co/tdnathmlenthusiast/cognitive-fatigue-detector)"
    )

    with gr.Tabs():

        # ── Tab 1: Webcam ─────────────────────────────────────────────────
        with gr.Tab("Live Webcam"):
            gr.Markdown(
                "Capture a photo from your webcam. "
                "Eye metrics estimated via OpenCV; temporal features use population medians. "
                "**For full precision, use the Upload CSV tab.**"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam — click to capture",
                        mirror_webcam=False,
                    )
                    analyze_webcam_btn = gr.Button("Analyze Photo", variant="primary")
                with gr.Column(scale=1):
                    # gr.Markdown instead of gr.HTML — fixes the schema crash
                    webcam_output_md    = gr.Markdown(label="Fatigue Analysis")
                    webcam_output_chart = gr.Image(
                        label="Feature Profile vs Population Median",
                        type="pil",
                        show_download_button=True,
                    )

            analyze_webcam_btn.click(
                fn=analyze_webcam,
                inputs=[webcam_input],
                outputs=[webcam_output_md, webcam_output_chart],
                api_name=False,
            )

        # ── Tab 2: CSV Upload ─────────────────────────────────────────────
        with gr.Tab("Upload CSV"):
            gr.Markdown(
                "Upload any GazeBase-format CSV. "
                "Runs the exact `extract_features()` pipeline — no approximation. "
                "Required columns: `n, x, y, val, dP, lab`"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    csv_input = gr.File(
                        file_types=[".csv"],
                        label="GazeBase CSV file",
                    )
                    analyze_csv_btn = gr.Button("Run Full Pipeline", variant="primary")
                    gr.Markdown(
                        "**Expected format:**\n"
                        "```\nn, x, y, val, dP, lab\n"
                        "1, 0.12, -0.04, 0, 412, 1\n"
                        "2, 0.13, -0.04, 0, 415, 2\n"
                        "...\n```"
                    )
                with gr.Column(scale=1):
                    # gr.Markdown instead of gr.HTML — fixes the schema crash
                    csv_output_md    = gr.Markdown(label="Fatigue Analysis")
                    csv_output_chart = gr.Image(
                        label="Feature Profile vs Population Median",
                        type="pil",
                        show_download_button=True,
                    )

            analyze_csv_btn.click(
                fn=analyze_csv,
                inputs=[csv_input],
                outputs=[csv_output_md, csv_output_chart],
                api_name=False,
            )

        # ── Tab 3: Scientific Results ─────────────────────────────────────
        with gr.Tab("Scientific Results"):
            gr.Markdown("### Evaluation metrics and training visualisations")
            # gr.Markdown instead of gr.HTML — fixes the schema crash
            gr.Markdown(value=build_results_html())

            gr.Markdown("---")
            gr.Markdown("#### Training Diagnostics")
            with gr.Row():
                gr.Image(value=load_plot("01_gazebase_training_curves.png"),
                         label="GazeBase training curves", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("02_gazebase_pred_vs_actual.png"),
                         label="Predicted vs actual fatigue score", type="pil",
                         show_download_button=True)

            gr.Markdown("#### LOSO Cross-Validation")
            with gr.Row():
                gr.Image(value=load_plot("03_loso_per_subject_auc_f1.png"),
                         label="AUC and F1 per subject", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("04_auc_distribution_boxplot.png"),
                         label="AUC distribution — all vs excl. S6", type="pil",
                         show_download_button=True)
            with gr.Row():
                gr.Image(value=load_plot("05_roc_curves_loso.png"),
                         label="ROC curves — all folds", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("07_drowsy_recall_per_subject.png"),
                         label="Drowsy recall per subject", type="pil",
                         show_download_button=True)

            gr.Markdown("#### Feature Analysis")
            with gr.Row():
                gr.Image(value=load_plot("06_feature_importance.png"),
                         label="Gradient-based feature importance", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("10_aggregate_comparison.png"),
                         label="Ablation: original (18) vs clean (16) features", type="pil",
                         show_download_button=True)
            with gr.Row():
                gr.Image(value=load_plot("08_clean_vs_orig_auc.png"),
                         label="AUC: original vs clean per subject", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("09_clean_vs_orig_f1.png"),
                         label="F1: original vs clean per subject", type="pil",
                         show_download_button=True)

            gr.Markdown("#### UMAP Latent Space")
            with gr.Row():
                gr.Image(value=load_plot("11_umap_fatigue_score.png"),
                         label="UMAP — fatigue score gradient", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("12_umap_task_type.png"),
                         label="UMAP — task type separation", type="pil",
                         show_download_button=True)
            with gr.Row():
                gr.Image(value=load_plot("13_umap_round_number.png"),
                         label="UMAP — round number (1→9)", type="pil",
                         show_download_button=True)
                gr.Image(value=load_plot("14_umap_session.png"),
                         label="UMAP — session (S1 vs S2)", type="pil",
                         show_download_button=True)

        # ── Tab 4: About ──────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

demo.queue()
demo.launch()