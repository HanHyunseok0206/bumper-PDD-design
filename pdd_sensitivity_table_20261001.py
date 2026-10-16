"""
pdd_sensitivity_table_20261001.py

목적: Case 4 DOE(888 trial, 불확실성 변수 U1~U3 포함) PDD 민감도 순위를 팀원 공유용 PDF 표로 출력.
      pdd_fourth_pass_20261001.py의 결과를 그대로 표로 렌더링.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

from pdd_fourth_pass_20261001 import VAR_COLS, NOISE_NAMES, load_clean, run_pdd_generic

HIGHLIGHT = "#2a78d6"   # 설계변수 상위(누적 85% 이내)
NOISE_COLOR = "#eb6834"  # 노이즈 변수
MUTED = "#52514e"
HEADER_BG = "#1a1a19"


def main():
    df = load_clean("doe_results_raw_20261001.csv")
    res = run_pdd_generic(df, VAR_COLS, "Case 4 DOE")

    rows = res["rows"]
    cum = 0.0
    table_rows = []
    for rank, (name, val) in enumerate(rows, start=1):
        cum += val
        table_rows.append([str(rank), name, f"{val:.4f}", f"{cum*100:.1f}%"])

    n_rows = len(table_rows)
    fig_h = 1.3 + 0.33 * n_rows
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axis("off")

    col_labels = ["순위", "변수", "민감도 지수", "누적"]
    tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            name = table_rows[r - 1][1]
            cum_pct = float(table_rows[r - 1][3].rstrip("%"))
            if name in NOISE_NAMES:
                cell.get_text().set_color(NOISE_COLOR)
            elif cum_pct <= 85.0:
                cell.get_text().set_color(HEADER_BG)
                if c == 1:
                    cell.get_text().set_fontweight("bold")
            else:
                cell.get_text().set_color(MUTED)
            cell.set_facecolor("#fcfcfb")

    fig.suptitle("PDD 민감도 순위 — Case 4 (N=888)", fontsize=13, color=HEADER_BG, y=0.995)
    fig.text(0.5, 0.01,
              f"홀드아웃 검증: m={res['best_n']}, S={res['best_y']} (상호작용 없음), "
              f"검증 R²={res['cv_mean']:.3f}±{(sum((r-res['cv_mean'])**2 for r in res['r2s'])/len(res['r2s']))**0.5:.3f}  |  "
              f"굵게/파란 계열=누적 85% 이내 설계변수, 주황=불확실성 변수(U1~U3)",
              ha="center", fontsize=7.5, color=MUTED)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = "pdd_sensitivity_ranking_20261001.pdf"
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"저장됨: {out_path}")


if __name__ == "__main__":
    main()
