
# X-Learner GUI Lab (Tkinter)

Generated 20250921_054900. This scaffold:
- Shows selected T1/T0 tickers and date range in the top bar
- Fetches data (yfinance) and renders charts, tables, and basic metrics per tab
- Embeds matplotlib figures in Tkinter
- Has Export & Copy bar on each tab
- Packages the whole project to ZIP (Topbar → '프로젝트 내보내기 ⬇')

Run:
```bash
pip install -r requirements.txt
python app.py
```

---

## 1) 설치 커맨드

```bash
pip install -U finance-datareader
```

> 패키지 이름은 하이픈 포함 **`finance-datareader`** 입니다. (대소문자 구분 없음)

---

## 2) `requirements.txt` (안전 호환 버전권장)

> TensorFlow 같은 무거운 DL 스택은 **옵션**으로 분리했습니다. 기본 실행(MLP/LGBM/XGB, 데이터 수집/시각화/GUI)은 아래 요구사항으로 충분합니다.

```
# --- Core scientific stack ---
numpy>=1.24,<3
pandas>=2.2,<3
matplotlib>=3.8,<4
scikit-learn>=1.4,<1.6
statsmodels>=0.14,<0.15
scipy>=1.11,<2

# --- Data sources ---
yfinance>=0.2.40,<0.3
finance-datareader>=0.9.90
pandas-datareader>=0.10,<0.11
fredapi>=0.5.0,<0.6
requests-cache>=1.2,<1.3

# --- Gradient boosting (선택, 기본 UI에 있음) ---
lightgbm>=4.3,<5
xgboost>=2.0,<3

# --- App/Utils ---
pyyaml>=6.0,<7
joblib>=1.3,<2
tqdm>=4.66,<5

# --- Tkinter는 OS에 내장(Python 배포) ---
# macOS: Python.org/Conda 배포에 포함. pip로 설치하지 않습니다.
```

### 선택: 딥러닝 스택(Apple Silicon, macOS) — `requirements-dl.txt`

TensorFlow를 쓰려면 **Python 3.10/3.11** 같은 호환 버전을 권장합니다(3.13 비권장). 아래는 Apple Silicon 기준입니다.

```
# macOS arm64 전용 권장(Apple Silicon)
tensorflow-macos==2.15.*; platform_system=="Darwin" and platform_machine=="arm64" and python_version<"3.12"
tensorflow-metal==1.1.*; platform_system=="Darwin" and platform_machine=="arm64" and python_version<"3.12"

# Keras는 TF에 포함되지만, 독립 사용 시 버전 정합 필요
keras>=2.15,<3; python_version<"3.12"
```

> Intel/Windows/Linux의 TF는 환경 제약이 까다롭습니다. 이 경우는 README의 환경 셋업 가이드를 따르세요.

---

## 3) `README.md` (업데이트판)

````markdown
# X-Learner GUI Lab

시계열/인과추론/금융 데이터를 활용해 **T1(경기민감 바스켓)** vs **T0(경기둔감 바스켓)**를 비교하고
**X-Learner**로 CATE/정책을 추정하는 Tkinter GUI 실험도구.

- 데이터 소스: **FinanceDataReader(우선)** → 실패 시 **yfinance** 폴백  
- 거시지표: **FRED API**(권장) → 일부는 yfinance 프록시(UUP, ^TNX/100, CL=F 등)  
- 모델: MLP/LGBM/XGB(기본), (옵션) CNN/LSTM/GRU/RNN(TensorFlow/Apple Silicon 권장)

---

## 빠른 시작

### 0) 파이썬 버전
- 권장: **Python 3.10 ~ 3.11**
- TF를 쓰지 않는다면 3.12도 가능하지만, 3.13은 일부 패키지 미호환.

### 1) 가상환경 생성(권장)

**Conda**
```bash
conda create -n xlearner python=3.11 -y
conda activate xlearner
````

**venv**

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 2) 의존성 설치

```bash
pip install -U pip wheel
pip install -r requirements.txt
```

> 한국/글로벌 가격계열을 안정적으로 받기 위해 **FinanceDataReader**를 기본 사용합니다.

```bash
pip install -U finance-datareader
```

### 3) (선택) 딥러닝 스택 (Apple Silicon, macOS)

```bash
# Python 3.10/3.11 권장
pip install -r requirements-dl.txt
```

### 4) FRED API 키(선택)

```bash
export FRED_API_KEY="YOUR_KEY"    # macOS/Linux
# set FRED_API_KEY=YOUR_KEY       # Windows
```

키가 없거나 FRED 호출 실패 시, 일부 지표(DGS10/DTWEXBGS/DCOILWTICO 등)는
yfinance 프록시(UUP, ^TNX/100, CL=F)를 **명시적으로 표시**하고 사용합니다.

### 5) 실행

```bash
python app.py
```

---

## 주요 기능

* **TopBar**: 적용/데이터받기/모두실행/중단/프로젝트 내보내기 + 전역 진행바
* **포트폴리오 탭**: T1/T0 티커/기간/주기/리밸/k-days/WF-splits/모델/하이퍼 설정
* **Tab0 데이터**:

  * 가격: FDR→yfinance 폴백, 동가중 로그수익률 생성
  * **Use Macro** 체크 시 거시지표 취득(FRED, yfinance 프록시) 및 **Macro Preview 표**
* **Tab1 μ-모델**: μ₁/μ₀ 예측, 검증 지표/곡선, 모델/하이퍼 표시
* **Tab2 Counterfactual**: D¹, D⁰ 산출/분포
* **Tab3 τ/e(X)**: τ₁/τ₀ 회귀 + 경향점수 분류, 디버그 로그/CSV, 분포/지표
* **Tab4 정책 평가**:

  * 정책 vs All-T1/All-T0 성과표(Sharpe/Sortino/MDD/Turnover/비용 반영/연도별 성과/포지션 히트맵)
  * 누적수익 곡선 + 드로다운 곡선 동시 표시

모든 탭에 **Export & Copy 바**(CSV/Markdown 복사, 그림 저장, 폴더 열기) 제공.

---

## 팁 & 트러블슈팅

* **macOS Tkinter**: Python.org 설치판/Conda는 Tk 포함. 화면 글꼴/렌더링이 어색하면 `defaults write -g NSRequiresAquaSystemAppearance -bool Yes` 등을 참고.
* **Matplotlib 백엔드**: 기본 TkAgg. 백엔드 문제 시 `MPLBACKEND=TkAgg` 환경변수로 강제.
* **ARM Mac에서 TF**: `tensorflow-macos + tensorflow-metal` 조합, Python 3.10/3.11 권장.
* **프록시 사용 표기**: 보고서/로그에 “FRED 불가 → yfinance 프록시”를 명시합니다.

---

## 요구사항 파일

* `requirements.txt` : 기본 실행에 필요한 의존성
* `requirements-dl.txt` : 선택 딥러닝 스택(Apple Silicon/TF)

---

## 라이선스

본 저장소 코드는 학술/사내 연구 목적 샘플입니다. 시장 데이터 사용은 각 공급자 ToS를 준수하세요.

```

---



2025-09-22 기준 X-Learner + AI 모델** 실험의 “전체 흐름” 정리

---

# 1) “현재 파이프라인 요약"

## 데이터·전처리 (Tab0: Step0)

* **자산 바스켓**

  * T1(경기민감): 예) XLY, XLI, XLF 등 (동일가중)
  * T0(경기둔감): 예) BTC-USD, ETH-USD (동일가중)
* **가격수집**

  * FDR 우선 → 실패 시 yfinance 폴백.
  * 보조지수: ^GSPC, ^VIX 등 필요 시 Fetch.
* **수익률**

  * 각 바스켓 종가 → 로그수익률 → **동일가중 평균**으로 `ret_T1`, `ret_T0`.
* **매크로 (옵션: Use Macro)**

  * FRED: INDPRO, CPIAUCSL, UNRATE, DGS10, DTWEXBGS, DCOILWTICO, NFCI (+발표지연 lag, 리샘플, 표준화)
  * yfinance 프록시: ^VIX, UUP, ^TNX/100, CL=F 등 (FRED 실패 시 표시·기록)
* **결과**

  * 누적수익(로그 누적), 최근 테이블, 매크로 프리뷰(활성 시) 저장/복사/Export.

## 결과모델 μ 학습 (Tab1: Step1)

* **목표**

  * $\mu_1(x) \approx \mathbb{E}[Y \mid X, T{=}1]$
  * $\mu_0(x) \approx \mathbb{E}[Y \mid X, T{=}0]$
* **구현**

  * 피처: `ret_T1`, `ret_T0`의 **시퀀스 랙 특성**(윈도우=seq\_window, 표준화 옵션) + (옵션) 매크로
  * 타깃: **k-일 포워드 수익률** 생성
  * 분할: 시계열 기반 Train/Test
  * 모델: **DL(CNN/LSTM/GRU/RNN)** 또는 **ML(LGBM/XGB/MLP)**
    (DL 환경 없으면 **폴백**을 로깅)
  * 출력: μ1/μ0 시계열, 유효성 지표(상관, RMSE), 예측곡선, 모델/하이퍼 로그

## Counterfactual 생성 (Tab2: Step2)

* **정의**

  * 처리군(주식) i에 대해 $D^1_i = Y_i - \hat{\mu}_0(X_i)$
  * 대조군(코인) j에 대해 $D^0_j = \hat{\mu}_1(X_j) - Y_j$
* **구현**

  * Step1의 μ 결과로 D¹/D⁰ 산출·정렬·분포 요약(표/히스토그램)

## 효과모델 τ + 경향점수 e(X) (Tab3: Step3)

* **목표**

  * $\tau_1(x) \approx \mathbb{E}[D^1 \mid X]$, $\tau_0(x) \approx \mathbb{E}[D^0 \mid X]$
  * $e(X)=P(T{=}1\mid X)$ (경향점수)
* **구현**

  * 피처: 동일한 랙 특성(옵션: 매크로 포함)
  * 회귀: τ₁/τ₀ 각각 **선택한 모델**로 학습
  * 분류: e(X) = 로지스틱/LGBM 등
  * **최종 CATE**: $\hat{\tau}(x) = e\cdot\hat{\tau}_0 + (1-e)\cdot\hat{\tau}_1$
  * 출력: τ/e 분포, 디버그 CSV, 상관(τ vs μ-gap) 등 지표, 학습/예측 시간 로깅

## 정책 평가 (Tab4: Step4)

* **정책 규칙(데모)**

  * 기본: $\hat{\mu}_1 > \hat{\mu}_0 \Rightarrow T1$ (또는 $\hat{\tau}>0 \Rightarrow T1$로 교체 가능)
* **백테스트**

  * 전략 수익률 vs All-T1 vs All-T0
  * **지표**: Sharpe/Sortino, MDD, Turnover, 비용 반영 성과, 연도별 성과, **레짐별 성과(VIX Low/High, 금리 Up/Down)**, 포지션 히트맵
  * **그래프**: 누적곡선 + **드로다운 곡선** 동시 표시
  * Export/Copy, 로그 저장

---

# 2) “추가로 더 개발해야 할 부분” — 연구·엔지니어링 체크리스트

## 인과추론·통계적 정확성

* [ ] **진짜 X-Learner 분리학습**:

  * 처리군 샘플로 μ₁, 대조군 샘플로 μ₀을 **각각** 학습하는 구조를 명확히(현재는 통합 피처/분할 중심).
  * τ₁은 **대조군 데이터로**(D¹ on control X), τ₀은 **처리군 데이터로**(D⁰ on treated X) 학습하도록 보장.
* [ ] **Cross-fitting / K-fold metalearning**: μ/τ/e 학습-예측 데이터 분리(오버피팅·편의 줄이기).
* [ ] **AIPW/DR(이중강건) 추정** 옵션: S/T/X-Learner와 비교 스위치.
* [ ] **공통지지(Overlap) 트리밍**: e(X)∈\[α,1−α] 필터의 효과 로그/리포트화.
* [ ] **누설(leakage) 방지**: 매크로 발표지연, 리샘플, 정규화는 **과거 정보만** 사용했는지 자동 점검.

## 모델·특성공학

* [ ] **특성 집합 모듈화**: Lag/MA/Vol/Drawdown/Regime dummy(저/고 VIX, 금리 방향) 등 플러그형.
* [ ] **DL 시퀀스 입력**: CNN/LSTM/GRU/RNN의 입력 텐서 reshape, 마스킹, 학습률 스케줄/조기종료 고도화.
* [ ] **Hyper-param 탐색**: Optuna/RandomSearch + WF-CV 스케줄.
* [ ] **모델 해석**: SHAP/Permutation importance로 μ/τ 중요 피처 시각화.

## 데이터/인프라

* [ ] **FDR→yfinance 폴백**의 상태 마킹(리포트에 “프록시 사용” 문구 자동 삽입).
* [ ] **FRED 키·콜 실패 시 재시도/캐시**(data/ 캐시) + 발표지연 계산 자동화.
* [ ] **프로젝트 Export**: 실행 설정/모델 가중치/결과물(표·그림·로그)을 zip으로 내보내기(버튼 이미 존재, 내용 강화).

## 백테스트·리스크

* [ ] **정확한 거래룰**: 리밸 주기/슬리피지(bps)/체크(시장가 체결 가정 vs 종가 체결) 일관화.
* [ ] **거래비용·Turnover 제약**: 포지션 스무딩, 신호 임계값(τ-threshold) 조절.
* [ ] **오프폴리시 평가**(IPS/DR)로 정책 가치 추정 보강(특히 실제 체결 불가 환경 대비).

---

# 3) “X-Learner + AI모델” 전체 흐름(권장 운영 시나리오)

아래는 **실전 정석 프로시저**를 코드·UI 단계에 정확히 매핑한 것입니다.

## 단계 A — 데이터 준비 (Tab0)

1. 기간/주기/리밸/k-days, 모델/하이퍼 설정 → **\[적용]**
2. **\[데이터 받기]**:

   * FDR→yfinance로 T1/T0 가격 수집 → `ret_T1`, `ret_T0`
   * (옵션) `Use Macro` 체크 시 FRED + 프록시 취득, **lag/표준화/일간화**
   * 공통 DatetimeIndex로 **정렬·정합**
   * 누적수익 그래프/테이블 확인

## 단계 B — μ 모델 학습 (Tab1)

3. 피처 $X$:

   * 랙 윈도우(예: 60)로 T1/T0 수익률 시퀀스, (옵션) 매크로도 **동일 시점 기준**으로 결합
   * 목표 $y$: **k-일 포워드**(누설 없게 shift)
4. 처리군/대조군 **분리 학습** 권장

   * $\mu_1$: (처리군) $X_{t}\to Y_{t+k}$
   * $\mu_0$: (대조군) $X_{t}\to Y_{t+k}$
   * 모델: LSTM/GRU/CNN/MLP/LGBM/XGB (환경에 맞게 자동 폴백)
5. 검증 로그/지표/곡선 확인, μ1/μ0 결과 저장

## 단계 C — Counterfactual (Tab2)

6. $D^1 = Y - \hat{\mu}_0(X)$, $D^0 = \hat{\mu}_1(X) - Y$ 산출, 분포 확인

   * **i는 T=1 샘플**, **j는 T=0 샘플**에 대해 계산(샘플 그룹 매칭 주의)

## 단계 D — τ/e 학습 (Tab3)

7. 피처 $X$ 재사용 (옵션 매크로 포함)
8. **X-Learner 규칙대로**:

   * $\tau_1(x)$: **대조군 데이터**(T=0)의 $D^1$에 대해 회귀
   * $\tau_0(x)$: **처리군 데이터**(T=1)의 $D^0$에 대해 회귀
   * $e(X)$: (T=1 vs 0) 경향점수 분류(로지스틱/LGBM 등)
9. 최종 $\hat{\tau}(x)=e\hat{\tau}_0+(1-e)\hat{\tau}_1$ 계산

   * 진단: τ와 (μ1−μ0) 상관, 분포, e(X) 평균/분산, 공통지지 트리밍 리포트

## 단계 E — 정책·백테스트 (Tab4)

10. **정책 규칙 선택**

    * A안(μ정책): $\hat{\mu}_1>\hat{\mu}_0 \Rightarrow T1$
    * B안(τ정책): $\hat{\tau}>0 \Rightarrow T1$  ※ 인과정책 권장
11. 포트폴리오 생성: $r_{strat}= I(T1)\cdot r_{T1} + (1-I(T1))\cdot r_{T0}$

    * 리밸/거래비용/슬리피지/Turnover 제약 반영
12. **리포트**

    * Sharpe/Sortino, **Turnover**, 비용 반영 성과, **연도별 성과**, **레짐별 성과(VIX Low/High, 금리 Up/Down)**
    * **누적곡선 + 드로다운 곡선**(2축), 포지션 히트맵
    * Export(표/마크다운/그림/zip)

---

## 보너스: 최소 수식 요약

* **μ-Learner**
  $\hat{\mu}_t(x)=\text{Regress}(Y \mid X, T=t)$, $t\in\{0,1\}$

* **X-Learner**
  $\hat{D}^1_i = Y_i - \hat{\mu}_0(X_i)$ for $i: T_i=1$
  $\hat{D}^0_j = \hat{\mu}_1(X_j) - Y_j$ for $j: T_j=0$
  $\hat{\tau}_1(x)=\text{Regress}(\hat{D}^1 \mid X)$ on **control** domain
  $\hat{\tau}_0(x)=\text{Regress}(\hat{D}^0 \mid X)$ on **treated** domain
  $\hat{e}(x)=P(T=1\mid X)$
  $\boxed{\hat{\tau}(x)=\hat{e}(x)\hat{\tau}_0(x) + (1-\hat{e}(x))\hat{\tau}_1(x)}$

---

### 정리

* **지금 상태**: 데이터 수집(FDR/yf), 매크로 옵션, μ/τ/e/정책 백테스트 end-to-end가 **동작**하고, UI/로그/Export까지 구성됨.
* **다음 단계**: *진짜* X-Learner의 **군별 학습/교차적합·DR·트리밍**을 엄격히 넣고, **매크로 누설 방지 자동검증**/하이퍼탐색/오프폴리시 평가/리포트 고도화까지 마무리.
* **운영 권장**: 실험은 “μ정책”과 “τ정책”을 **둘 다** 돌려 비교하고, **레짐별 성과**와 **비용 반영 성과**를 핵심 KPI로 관리


