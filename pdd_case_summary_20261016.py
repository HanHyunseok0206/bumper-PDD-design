"""
pdd_case_summary_20261016.py

목적: 2026-10-13 지도교수 미팅 피드백 반영 — PDD 결과표를 "n/y"가 아니라 기존 PDD 논문
      표기(m=차수, S=상호작용 차수/S-variate 절단)로 통일하고, 케이스(Case)별로 설계변수
      개수/불확실성 변수 개수를 분리해서 명시한 요약표. 데이터는 전부 과거 실행 결과를 그대로
      재인용(pdd_third_pass_20260929.py / pdd_fourth_pass_20261001.py / pdd_fifth_pass_20261007.py
      / pdd_sixth_pass_20261014.py 실행 로그, 숫자 재계산 아님 — 재계산해도 완전히 같은 값이
      나와야 정상이고 실제로 과거 캐시된 실행 로그와 대조해 확인함).

      주의: Case 6(6차, 집중 재해석)은 Case 1~5와 변수 구성이 다름(설계변수 19개 전체가
      아니라 상위 6개만, 나머지 13개는 고정) — 같은 표에 넣되 "비교 불가, 참고용"으로 표시.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False

# (case, 날짜, N, 설계변수 수, 불확실성변수 수, m, S, 학습R2, CV R2 평균, CV R2 표준편차, gap, top 변수)
CASES = [
    ("Case 1", "2026-09-22", 90, 19, 0, 1, 1, 0.477, 0.002, 0.296, 0.475,
     "B9, B2, B6, B8, B3 (불안정 — 참고용)"),
    ("Case 2", "2026-09-23", 603, 19, 0, 3, 1, 0.424, 0.274, 0.088, 0.150,
     "B9, B4, B8, C4, C2"),
    ("Case 3", "2026-09-29", 4949, 19, 0, 6, 1, 0.424, 0.427, 0.047, -0.003,
     "B4, B9, C2, C4, B1"),
    ("Case 4", "2026-10-01", 888, 19, 3, 5, 1, 0.553, 0.383, 0.040, 0.169,
     "B4, B9, C2, C4, B1 (+U1 2.0%)"),
    ("Case 5", "2026-10-07", 2932, 19, 3, 5, 1, 0.285, 0.276, 0.096, 0.009,
     "B9, B4, C2, B6, C4 (+U1 1.6%)"),
    ("Case 6*", "2026-10-14", 778, 6, 3, 4, 1, 0.628, 0.591, 0.030, 0.037,
     "B6 58%, B9 18%, U1 15% (집중 9변수, 비교 주의)"),
]

col_labels = ["Case", "날짜", "N", "설계\n변수", "불확실성\n변수", "m\n(차수)", "S\n(상호작용)",
              "학습\nR²", "검증 R²\n(10-seed 평균±σ)", "gap", "상위 변수(기여율)"]

table_rows = []
for case, date, N, ndes, nunc, m, S, tr, cvm, cvs, gap, top in CASES:
    table_rows.append([case, date, str(N), str(ndes), str(nunc), str(m), str(S),
                        f"{tr:.3f}", f"{cvm:.3f}±{cvs:.3f}", f"{gap:+.3f}", top])

fig, ax = plt.subplots(figsize=(14, 1.1 + 0.55 * len(table_rows)))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axis("off")

tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)

col_widths = [0.06, 0.08, 0.05, 0.06, 0.08, 0.06, 0.08, 0.06, 0.13, 0.06, 0.28]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d8d7d2")
    cell.set_width(col_widths[c])
    if r == 0:
        cell.set_facecolor("#1a1a19")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#fcfcfb")
        cell.get_text().set_color("#1a1a19")
        if c == 0 and "*" in table_rows[r - 1][0]:
            cell.get_text().set_color("#eb6834")

fig.suptitle("PDD 민감도 스크리닝 — Case별 요약 (m=차수, S=상호작용 차수)",
              fontsize=13, color="#1a1a19", y=0.98)
fig.text(0.5, 0.01,
          "gap = 학습R² − 검증R² 평균 (클수록 과적합 의심). Case 1~5: 19개 설계변수(+Case4~5는 불확실성 변수 U1~U3) 동시 적합.\n"
          "*Case 6: 상위 6개 설계변수만 남기고 13개 고정한 집중 재해석 — N과 변수 범위가 달라 Case 1~5와 직접 비교 불가, 참고용.",
          ha="center", fontsize=8, color="#52514e")

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
out_path = "pdd_case_summary_20261016.pdf"
plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"저장됨: {out_path}")
