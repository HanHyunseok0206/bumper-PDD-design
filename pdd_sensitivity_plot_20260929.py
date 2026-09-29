"""
pdd_sensitivity_plot_20260929.py

목적: 1차/2차/3차 PDD 민감도 지수를 3패널 막대그래프로 비교. pdd_sensitivity_plot_20260923.py의
      2패널 버전을 3차 데이터 추가로 확장.

강조 기준: 표본이 충분히 큰 2차(N=603)·3차(N=4949) 둘 다에서 top-5에 든 변수만 강조.
      1차(N=90)는 검증 R2가 거의 0(과적합)이라 신뢰도가 낮아 강조 기준에서 제외하되,
      비교를 위해 그림에는 그대로 표시.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

import pandas as pd

from pdd_second_pass_20260923 import run_pdd, load_second_pass

HIGHLIGHT = "#2a78d6"   # dataviz 팔레트 categorical slot 1 (blue)
MUTED = "#c2c1bc"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load_df1():
    df1 = pd.read_csv("doe_results_raw_20260922.csv")
    df1 = df1[df1["SEA"] != 0].copy()
    median_ie1 = df1["Internal_Energy"].median()
    return df1[df1["Internal_Energy"] > median_ie1 * 0.5].reset_index(drop=True)


def plot_panel(ax, rows, highlight_names, title, n_total):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [HIGHLIGHT if n in highlight_names else MUTED for n in names]

    y_pos = range(len(names))
    ax.barh(y_pos, vals, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlabel("민감도 지수\n(SEA 분산 기여율)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(f"{title} (N={n_total})", fontsize=11.5, color=TEXT_PRIMARY, loc="left")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7d2")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    xmax = max(vals) if vals else 1
    for y, v in zip(y_pos, vals):
        ax.text(v + xmax * 0.02, y, f"{v:.3f}", va="center", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xlim(0, xmax * 1.18)


def main():
    df1 = load_df1()
    df2 = load_second_pass("doe_results_raw_20260923.csv")
    df3 = load_second_pass("doe_results_raw_20260929.csv")

    res1 = run_pdd(df1, "1차 DOE (2026-09-22)")
    res2 = run_pdd(df2, "2차 DOE (2026-09-23)")
    res3 = run_pdd(df3, "3차 DOE (2026-09-29)")

    top2 = {name for name, _ in res2["rows"][:5]}
    top3 = {name for name, _ in res3["rows"][:5]}
    shared = top2 & top3  # 표본이 충분히 큰 2차·3차 둘 다에서 top-5인 변수만 강조 (1차는 신뢰도 낮아 제외)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    plot_panel(axes[0], res1["rows"], shared, "1차 DOE (ISO 공차 범위)", len(df1))
    plot_panel(axes[1], res2["rows"], shared, "2차 DOE (범위 확대)", len(df2))
    plot_panel(axes[2], res3["rows"], shared, "3차 DOE (표본 대폭 확대)", len(df3))

    fig.suptitle("PDD 민감도 분석: 1차 vs 2차 vs 3차 DOE 비교", fontsize=14.5, color=TEXT_PRIMARY, y=1.03)
    fig.text(0.5, -0.03,
              f"파란색 = 표본이 충분히 큰 2차·3차 모두 상위 5위 안에 든 변수({', '.join(sorted(shared))}) — "
              "데이터가 늘어도 일관되게 중요했던 신호 (1차는 검증 R²≈0으로 신뢰도가 낮아 강조 기준에서 제외)",
              ha="center", fontsize=9, color=TEXT_SECONDARY)

    plt.tight_layout()
    out_path = "pdd_sensitivity_comparison_20260929.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장됨: {out_path}")


if __name__ == "__main__":
    main()
