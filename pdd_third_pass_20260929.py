"""
pdd_third_pass_20260929.py

목적: 팀원이 전달한 3차 DOE 결과(doe_results_raw_20260929.csv, 4949 trial)로 PDD 민감도 분석을
      재실행하고 1차(90개)/2차(603개)와 비교. 함수는 pdd_first_pass_20260922.py와 동일

      3차 데이터 특징:
        - 표본 4949개 (2차 대비 약 8배, 1차 대비 약 55배)
        - 설계변수 변동폭이 B1~C8은 공칭값 대비 ~6.6%
"""

import numpy as np
import pandas as pd

from pdd_first_pass_20260922 import VAR_COLS, PDD, min_max_scale
from pdd_second_pass_20260923 import run_pdd, load_second_pass


def load_df1():
    df1 = pd.read_csv("doe_results_raw_20260922.csv")
    df1 = df1[df1["SEA"] != 0].copy()
    median_ie1 = df1["Internal_Energy"].median()
    return df1[df1["Internal_Energy"] > median_ie1 * 0.5].reset_index(drop=True)


def main():
    df1 = load_df1()
    df2 = load_second_pass("doe_results_raw_20260923.csv")
    df3 = load_second_pass("doe_results_raw_20260929.csv")  

    res1 = run_pdd(df1, "1차 DOE (2026-09-22, N=90)")
    res2 = run_pdd(df2, "2차 DOE (2026-09-23, N=603)")
    res3 = run_pdd(df3, "3차 DOE (2026-09-29, N=4949)")

    print(f"\n{'='*70}\n1/2/3차 비교 요약\n{'='*70}")
    for label, res in [("1차", res1), ("2차", res2), ("3차", res3)]:
        print(f"{label}: N={res['N']:5d}, k={res['k']:3d}(k/N={res['k']/res['N']:.4f}), "
              f"학습R2={res['train_r2']:.3f}, CV R2평균={res['cv_mean']:.3f}"
              f"±{np.std(res['r2s']):.3f}, gap={res['gap']:.3f} (n={res['best_n']},y={res['best_y']})")

    top1 = {name for name, _ in res1["rows"][:5]}
    top2 = {name for name, _ in res2["rows"][:5]}
    top3 = {name for name, _ in res3["rows"][:5]}
    print(f"\n1차 top5: {top1}")
    print(f"2차 top5: {top2}")
    print(f"3차 top5: {top3}")
    print(f"2차∩3차: {top2 & top3}")
    print(f"1차∩2차∩3차: {top1 & top2 & top3}")


if __name__ == "__main__":
    main()
