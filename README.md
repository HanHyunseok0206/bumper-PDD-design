# bumper-PDD-design

종합설계 주제인 "PDD 기반의 surrogate modeling을 이용한 범퍼 강건 설계"에서 쓰는 코드 저장소. 카티아로 만든 3D 범퍼 형상에 대해 SEA(비에너지흡수율)를 최적화하기 전에, 소수의 해석 결과로 응답을 근사하는 PDD(Polynomial Dimensional Decomposition) 서로게이트를 먼저 검증하는 단계.

## 파일

실제 파이프라인에서 쓰는 노트북(순서대로):

- `DOE_sampling.ipynb`: 실험계획법(DOE) 샘플 생성
- `PDD_bumper_SEA.ipynb`: PDD 기반 민감도 분석
- `Kriging_bumper_SEA.ipynb`: 크리깅 서로게이트 + 강건 최적화

검증·베이스라인·비교용 노트북은 `validation_and_benchmarks/`에 따로 모아둠:

- `validation_and_benchmarks/PDD_Legendre_ver4.ipynb`: 르장드르 다항식 기반 PDD 구현체 검증(Ishigami 벤치마크)
- `validation_and_benchmarks/PDD_vs_Kriging_benchmark.ipynb`: PDD vs 크리깅 정확도 비교
- `validation_and_benchmarks/Kriging_pipeline_demo.ipynb`: 크리깅 파이프라인 합성 데이터 검증

연구 진행 상황·컨벤션은 `GUIDELINE.md`, 논문/발표자료 초안은 `PAPER.md` 참고.

## 노트북 구성 (`validation_and_benchmarks/PDD_Legendre_ver4.ipynb`)

- `basis(x, a)`: 표준 르장드르 다항식을 점화식으로 계산 (0~a차)
- `PDD(x, n, y)`: 입력 `x`(dim × N)에 대해 1변수(단독), 2변수, 3변수 상호작용 항까지 포함한 PDD 기저 행렬을 생성. `n`은 항의 최대 차수, `y`는 고려할 상호작용 변수 개수(1=단독항만, 2=2변수 교호작용까지, 3=3변수까지)
- `min_max_scale` / `theoretical_scale`: 입력을 르장드르 다항식이 정의되는 `[-1, 1]` 구간으로 스케일링. 변수의 실제 정의역(설계변수 LB/UB 등)을 알고 있으면 `theoretical_scale`을 쓰는 게 맞고, 모를 때만 `min_max_scale`을 fallback으로 사용
- `get_sobol2(Ci, mapping, exp_input)`: PDD 계수로부터 각 변수/상호작용의 분산 기여도(Sobol 민감도 지수)를 계산. 지금 쓰는 르장드르 기저가 정규직교가 아니라서 계수 제곱합이 아니라 항별 예측값의 경험적 분산으로 계산함
- `find_optimal_degree(...)`: 데이터를 학습/검증으로 나눠서 검증셋 R²가 더 이상 개선되지 않을 때까지 차수(n)와 상호작용 차수(y)를 늘려가며 탐색. 학습 데이터로만 R²를 재면 차수를 올릴수록 항상 좋아 보여서 과적합되기 때문에 검증셋 기준으로 바꿈

Ishigami 함수(3변수 벤치마크)로 이론적 Sobol 지수와 대조해서 위 파이프라인이 맞게 동작하는지 확인해둔 상태.

## 사용법

1. 노트북 상단 셀에서 `input`(dim × N 행렬), `output`(N,) 형태로 데이터를 준비. 파일에서 불러올 경우 셀 안에 주석 처리된 `np.loadtxt` 예시 참고
2. 변수별 실제 정의역을 알고 있다면 `theoretical_scale(input, domain_min, domain_max)`로 `[-1, 1]`로 스케일링
3. `find_optimal_degree(input, output)`로 적절한 (n, y) 탐색
4. `PDD(input, n, y)`로 기저 행렬 생성 → `np.linalg.pinv`로 계수 추정
5. `get_sobol2`로 민감도 지수 확인, 필요하면 기여도 낮은 변수는 최적화 단계에서 제외

## 의존성

- numpy, pandas

## 진행 상황 / 다음 단계

- [x] Legendre 기반 PDD 및 Sobol 지수 계산 로직 검증 (Ishigami 벤치마크)
- [x] 학습/검증 분리 기반 차수 선택으로 변경 (과적합 방지)
- [ ] 카티아 범퍼 모델 설계변수 + FE 해석(SEA) 결과로 실데이터 적용
- [ ] PDD 계수 기반 mean/variance로 강건 최적화(RDO) 목적함수 구성
- [ ] 최적점 카티아/FE 재해석으로 서로게이트 검증
