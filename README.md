
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

필요하시면, 위 내용을 바탕으로 **`core/fetchers.py`의 FDR 우선 로더**(이미 설명드린 `fetch_prices_prefer_fdr`)와, `tab_data.py`에서 해당 유틸을 호출하도록 바꾼 **패치 파일**도 만들어 드릴게요.
::contentReference[oaicite:0]{index=0}
```

