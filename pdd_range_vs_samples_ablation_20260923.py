"""
pdd_range_vs_samples_ablation_20260923.py

목적: 2차 DOE(N=603, 범위 확대)가 1차(N=90, ISO 공차 범위)보다 PDD 검증 R2가 좋아진 게
      "표본이 늘어서"인지 "범위가 넓어져서"인지 분리해서 확인.

방법: 두 요인이 동시에 바뀌어서(confounded) 직접 분리는 안 되지만, 2차 데이터(넓은 범위)에서
      표본을 무작위로 서브샘플링하면 "범위는 넓고 표본만 적은" 조건을 만들 수 있음. 이걸
      1차(범위 좁고 표본도 적음) 및 2차 전체(범위 넓고 표본도 많음)와 비교하면:

        1차 실측 (N=90,  좁은 범위) -> 범위 효과 없이 표본만 적을 때
        2차 서브샘플 (N=90, 넓은 범위) -> 표본은 같은데 범위만 넓을 때  [범위 효과 분리]
        2차 서브샘플 (N=150~400, 넓은 범위) -> 넓은 범위에서 표본이 늘 때 [표본 효과 분리]
        2차 전체 (N=603, 넓은 범위) -> 최종 결과

      주의: X가 바뀌면 그에 대응하는 실제 FE 결과(Y)가 새로 필요하므로 "표본은 많은데 범위는
      좁은" 조건은 이 두 데이터셋만으로는 만들 수 없음(가상의 Y를 만들 수 없기 때문) — 그래서
      위 3가지 비교만 가능함.
"""

import numpy as np
import pandas as pd

from pdd_first_pass_20260922 import VAR_COLS, find_optimal_degree, min_max_scale


def load_df1():
    df1 = pd.read_csv("doe_results_raw_20260922.csv")
    df1 = df1[df1["SEA"] != 0].copy()
    median_ie1 = df1["Internal_Energy"].median()
    return df1[df1["Internal_Energy"] > median_ie1 * 0.5].reset_index(drop=True)


def cv_r2_stats(X_scaled, Y, n_seeds=10, max_n=6, max_y=2):
    r2s = [float(find_optimal_degree(X_scaled, Y, max_n=max_n, max_y=max_y, seed=s)[2])
           for s in range(n_seeds)]
    return r2s


def eval_full_dataset(df, label):
    X_raw = df[VAR_COLS].to_numpy().T
    Y = df["SEA"].to_numpy()
    X_scaled = min_max_scale(X_raw)
    r2s = cv_r2_stats(X_scaled, Y)
    print(f"{label:35s} N={len(df):4d}  CV R2 평균={np.mean(r2s):.3f} (표준편차 {np.std(r2s):.3f})")
    return r2s


def eval_subsample(df, n_sub, n_repeats=5, sub_seed_base=1000):
    all_r2s = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(sub_seed_base + rep)
        idx = rng.choice(len(df), size=n_sub, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
        X_raw = df_sub[VAR_COLS].to_numpy().T
        Y = df_sub["SEA"].to_numpy()
        X_scaled = min_max_scale(X_raw)
        r2s = cv_r2_stats(X_scaled, Y, n_seeds=5)  # 서브샘플 반복이 많아 시드는 5개로 축소
        all_r2s.extend(r2s)
    return all_r2s


def main():
    df1 = load_df1()
    df2 = pd.read_csv("doe_results_raw_20260923.csv")

    print("=" * 70)
    print("기준점")
    print("=" * 70)
    r2_narrow_n90 = eval_full_dataset(df1, "1차 실측 (N=90, 좁은 범위)")
    r2_wide_full = eval_full_dataset(df2, "2차 실측 (N=603, 넓은 범위)")

    print("\n" + "=" * 70)
    print("2차(넓은 범위) 데이터에서 표본 수만 바꿔가며 서브샘플링 (범위 효과 통제)")
    print("=" * 70)
    for n_sub in [90, 150, 250, 400]:
        r2s = eval_subsample(df2, n_sub, n_repeats=5)
        print(f"넓은 범위, N={n_sub:4d} (5회 서브샘플x5-seed={len(r2s)}개)  "
              f"CV R2 평균={np.mean(r2s):.3f} (표준편차 {np.std(r2s):.3f})")

    print("\n" + "=" * 70)
    print("해석용 요약")
    print("=" * 70)
    print(f"[범위 효과]  좁은 범위 N=90: {np.mean(r2_narrow_n90):.3f}  vs  "
          f"넓은 범위 N=90(서브샘플): 위 표 참고  -> 같은 N에서 범위만 바꾼 차이")
    print(f"[표본 효과]  넓은 범위 N=90(서브샘플) -> N=603(전체): {np.mean(r2_wide_full):.3f}  "
          f"-> 같은 범위에서 표본만 늘렸을 때 차이")


if __name__ == "__main__":
    main()
