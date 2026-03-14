import numpy as np
import torch
from huggingface_hub import hf_hub_download

from model import GazeModel

REPO_ID = "tdnathmlenthusiast/cognitive-fatigue-detector"

# ── Global state (loaded once at startup) ──────────────────────────────────
_model = None
_scaler_mean = None
_scaler_std = None
_feature_cols = None
_pop_stats = None
_loso_results = None
_stage1_test = None


def load_checkpoint():
    """
    Download and initialise the model checkpoint from HuggingFace Hub.
    Called once at app startup; results stored in module-level globals.
    """
    global _model, _scaler_mean, _scaler_std, _feature_cols
    global _pop_stats, _loso_results, _stage1_test

    ck_path = hf_hub_download(repo_id=REPO_ID, filename="gaze_clean_checkpoint.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    _model = GazeModel(ck["model_config"]["in_dim"])
    _model.load_state_dict(ck["model_state_dict"])
    _model.eval()

    _scaler_mean  = np.array(ck["scaler_mean"], dtype=np.float32)
    _scaler_std   = np.array(ck["scaler_std"],  dtype=np.float32)
    _feature_cols = ck["feature_cols"]
    _pop_stats    = ck.get("population_stats", {})
    _loso_results = ck.get("loso_clean_results", [])
    _stage1_test  = ck.get("stage1_test", {})

    print(f"Model loaded — {len(_feature_cols)} features, "
          f"R²={_stage1_test.get('r2', '?'):.3f}, "
          f"LOSO AUC≈{np.mean([r['auc'] for r in _loso_results]):.3f}")


def is_loaded() -> bool:
    return _model is not None


def get_population_stats() -> dict:
    return _pop_stats or {}


def get_loso_results() -> list:
    return _loso_results or []


def get_stage1_metrics() -> dict:
    return _stage1_test or {}


def predict(features: dict) -> float:
    """
    Normalise a feature dict and return a fatigue z-score.
    Uses the scaler fitted on the GazeBase training split.
    """
    feat_vec = np.array(
        [float(features.get(col, 0.0)) for col in _feature_cols],
        dtype=np.float32,
    )
    # Replace NaN/Inf with 0 before normalising
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    feat_norm = (feat_vec - _scaler_mean) / (_scaler_std + 1e-8)

    with torch.no_grad():
        x = torch.tensor(feat_norm).unsqueeze(0)
        score = float(_model(x).item())

    return score


def score_to_insights(fatigue_score: float, features: dict) -> dict:
    """
    Convert the raw z-score into user-facing insight strings.
    fatigue_score is z-scored; typical range is roughly −2.5 to +2.5.
    """
    fatigue_pct = int(np.clip((fatigue_score + 2.5) / 5.0 * 100, 0, 100))

    # Mental state
    if fatigue_pct < 30:
        state, state_color, state_desc = "Alert", "#27AE60", "Focused and sharp"
    elif fatigue_pct < 55:
        state, state_color, state_desc = "Mild Fatigue", "#F39C12", "Attention beginning to drift"
    elif fatigue_pct < 75:
        state, state_color, state_desc = "Moderate Fatigue", "#E67E22", "Reaction time slowing"
    else:
        state, state_color, state_desc = "High Fatigue", "#E74C3C", "Performance significantly impaired"

    # Blink health (literature: 15–20 blinks/min is healthy)
    blink_rate = float(features.get("blink_rate_pm", 0))
    if blink_rate < 8:
        blink_msg, blink_color = f"{blink_rate:.0f}/min — severely reduced", "#E74C3C"
    elif blink_rate < 15:
        blink_msg, blink_color = f"{blink_rate:.0f}/min — below normal (15–20/min)", "#F39C12"
    elif blink_rate <= 20:
        blink_msg, blink_color = f"{blink_rate:.0f}/min — healthy range", "#27AE60"
    else:
        blink_msg, blink_color = f"{blink_rate:.0f}/min — elevated (possible irritation)", "#3498DB"

    # Eye strain: saccade velocity vs population baseline 280 deg/s
    sac_vel  = float(features.get("mean_sac_vel", 280))
    baseline = 280.0
    strain_pct = max(0, int((1 - sac_vel / baseline) * 100)) if baseline > 0 else 0
    if strain_pct < 20:
        strain_msg, strain_color = "Low — velocity within normal range", "#27AE60"
    elif strain_pct < 45:
        strain_msg, strain_color = f"Moderate — {strain_pct}% below baseline", "#F39C12"
    else:
        strain_msg, strain_color = f"High — {strain_pct}% velocity drop", "#E74C3C"

    # Break advisor
    if fatigue_pct >= 75:
        break_msg, break_color = "Take a 5-min break RIGHT NOW", "#E74C3C"
    elif fatigue_pct >= 55:
        break_msg, break_color = "Short break recommended within 10 min", "#E67E22"
    elif fatigue_pct >= 35:
        break_msg, break_color = "Consider a break in the next 20 min", "#F39C12"
    else:
        break_msg, break_color = "No break needed — keep going", "#27AE60"

    return {
        "fatigue_pct":  fatigue_pct,
        "fatigue_score": round(fatigue_score, 3),
        "state":        state,
        "state_color":  state_color,
        "state_desc":   state_desc,
        "blink_msg":    blink_msg,
        "blink_color":  blink_color,
        "strain_msg":   strain_msg,
        "strain_color": strain_color,
        "break_msg":    break_msg,
        "break_color":  break_color,
    }


def build_insights_html(insights: dict, source_label: str = "", disclaimer: str = "") -> str:
    """Render the four insight cards as an HTML string for gr.HTML."""
    fp  = insights["fatigue_pct"]
    bar_color = insights["state_color"]

    # Gradient position for the bar fill
    bar_style = (
        f"background: linear-gradient(90deg, #27AE60, #F39C12, #E74C3C); "
        f"width: {fp}%; height: 12px; border-radius: 99px;"
    )

    disclaimer_block = (
        f'<div style="background:#FFF3CD;border-left:3px solid #F39C12;'
        f'padding:8px 12px;border-radius:4px;font-size:12px;'
        f'color:#856404;margin-bottom:12px;">'
        f'{disclaimer}</div>'
    ) if disclaimer else ""

    source_block = (
        f'<div style="font-size:11px;color:#888;margin-bottom:4px;">{source_label}</div>'
    ) if source_label else ""

    html = f"""
<div style="font-family:sans-serif;max-width:600px">
  {disclaimer_block}
  {source_block}

  <!-- Score gauge -->
  <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:10px;
              padding:16px 20px;margin-bottom:12px;">
    <div style="display:flex;align-items:center;justify-content:space-between;
                margin-bottom:10px;">
      <div>
        <span style="font-size:32px;font-weight:600;color:{bar_color};">{fp}%</span>
        <span style="font-size:13px;color:#666;margin-left:8px;">fatigue level</span>
      </div>
      <span style="background:{bar_color};color:#fff;padding:5px 14px;
                   border-radius:20px;font-size:13px;font-weight:500;">
        {insights["state"]}
      </span>
    </div>
    <div style="background:#e9ecef;border-radius:99px;height:12px;overflow:hidden;">
      <div style="{bar_style}"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:4px;">
      <span style="font-size:11px;color:#aaa;">Alert</span>
      <span style="font-size:11px;color:#aaa;">Fatigued</span>
    </div>
    <div style="font-size:12px;color:#666;margin-top:8px;">
      {insights["state_desc"]} &nbsp;·&nbsp; z-score: {insights["fatigue_score"]}
    </div>
  </div>

  <!-- Insight cards grid -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">

    <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;
                padding:12px 14px;">
      <div style="font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
        Mental State
      </div>
      <div style="font-size:15px;font-weight:500;color:{insights["state_color"]};">
        {insights["state"]}
      </div>
      <div style="font-size:12px;color:#666;margin-top:3px;">{insights["state_desc"]}</div>
    </div>

    <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;
                padding:12px 14px;">
      <div style="font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
        Blink Health
      </div>
      <div style="font-size:15px;font-weight:500;color:{insights["blink_color"]};">
        {insights["blink_msg"].split("—")[0].strip()}
      </div>
      <div style="font-size:12px;color:#666;margin-top:3px;">
        {"— ".join(insights["blink_msg"].split("—")[1:]).strip() if "—" in insights["blink_msg"] else ""}
      </div>
    </div>

    <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;
                padding:12px 14px;">
      <div style="font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
        Eye Strain
      </div>
      <div style="font-size:15px;font-weight:500;color:{insights["strain_color"]};">
        {insights["strain_msg"].split("—")[0].strip()}
      </div>
      <div style="font-size:12px;color:#666;margin-top:3px;">
        Saccade velocity vs 280 deg/s baseline
      </div>
    </div>

    <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;
                padding:12px 14px;">
      <div style="font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
        Break Advisor
      </div>
      <div style="font-size:15px;font-weight:500;color:{insights["break_color"]};">
        {insights["break_msg"]}
      </div>
      <div style="font-size:12px;color:#666;margin-top:3px;">
        Based on current fatigue level
      </div>
    </div>

  </div>
</div>
"""
    return html
