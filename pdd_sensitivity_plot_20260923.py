"""
pdd_sensitivity_plot_20260923.py

목적: pdd_second_pass_20260923.py가 계산한 1차/2차 PDD 민감도 지수(SEA 분산 기여율)를
      발표/보고서용 막대그래프로 시각화. 1차·2차 둘 다에서 상위 5위 안에 들었던 변수(B9, B8)를
      강조색으로, 나머지는 회색으로 표시해 "데이터가 바뀌어도 살아남은 신호"를 한눈에 보여줌.

색상: dataviz 스킬의 검증된 기본 팔레트(references/palette.md)에서 강조=blue(#2a78d6),
      비강조=중립 회색을 사용. 두 색만 쓰는 단순 대비라 별도 CVD 검증 없이도 안전.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

from pdd_second_pass_20260923 import run_pdd, load_second_pass
import pandas as pd

HIGHLIGHT = "#2a78d6"   # dataviz 팔레트 categorical slot 1 (blue)
MUTED = "#c2c1bc"       # 비강조 막대
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
    ax.set_yticklabels(names, fontsize=9, color=TEXT_PRIMARY)
    ax.invert_yaxis()  # 1위가 맨 위로
    ax.set_xlabel("민감도 지수 (SEA 분산 기여율)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(f"{title} (N={n_total})", fontsize=12, color=TEXT_PRIMARY, loc="left")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7d2")
    ax.tick_params(colors=TEXT_SECONDARY)

    # 값 라벨 (막대 끝에 직접 표기)
    xmax = max(vals) if vals else 1
    for y, v in zip(y_pos, vals):
        ax.text(v + xmax * 0.015, y, f"{v:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, xmax * 1.15)


def main():
    df1 = load_df1()
    df2 = load_second_pass("doe_results_raw_20260923.csv")

    res1 = run_pdd(df1, "1차 DOE (2026-09-22)")
    res2 = run_pdd(df2, "2차 DOE (2026-09-23)")

    top1 = {name for name, _ in res1["rows"][:5]}
    top2 = {name for name, _ in res2["rows"][:5]}
    shared = top1 & top2  # 두 데이터셋 모두에서 top-5 안에 든 변수만 강조

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    plot_panel(axes[0], res1["rows"], shared, "1차 DOE (ISO 공차 범위)", len(df1))
    plot_panel(axes[1], res2["rows"], shared, "2차 DOE (확대 범위)", len(df2))

    fig.suptitle("PDD 민감도 분석: 1차 vs 2차 DOE 비교", fontsize=14, color=TEXT_PRIMARY, y=1.02)
    fig.text(0.5, -0.02,
              f"파란색 = 두 데이터셋 모두 상위 5위 안에 든 변수({', '.join(sorted(shared))}) — "
              "데이터가 바뀌어도 일관되게 중요했던 신호",
              ha="center", fontsize=9, color=TEXT_SECONDARY)

    plt.tight_layout()
    out_path = "pdd_sensitivity_comparison_20260923.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n저장됨: {out_path}")


if __name__ == "__main__":
    main()
