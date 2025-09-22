# core/model_selector.py
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    LGBMRegressor = None
    LGBMClassifier = None

# NEW
try:
    from core.models_dl import KerasSeqRegressor, TF_AVAILABLE
except Exception:
    KerasSeqRegressor = None
    TF_AVAILABLE = False


def _mlp_fallback(kind: str, ctx: Dict[str, Any]) -> MLPRegressor:
    lr = float(ctx.get("lr", 1e-3))
    epochs = int(ctx.get("epochs", 20))
    batch = int(ctx.get("batch", 64))
    hidden = (128, 64) if kind in {"LSTM","GRU"} else (64, 32) if kind == "RNN" else (256,128,64)
    m = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        learning_rate_init=lr, max_iter=max(100, epochs*10), batch_size=batch,
        random_state=42, verbose=False
    )
    # 폴백 표시용 플래그 (로그에 사용)
    setattr(m, "_fallback_from", kind)
    return m


def get_regressor(kind: str, ctx: Dict[str, Any]):
    kind = (kind or "").upper()

    # DL 계열은 Keras 사용 (가능하면)
    if kind in {"LSTM","GRU","RNN","CNN"}:
        if TF_AVAILABLE and KerasSeqRegressor is not None:
            return KerasSeqRegressor(kind, ctx)
        else:
            return _mlp_fallback(kind, ctx)

    # 트리
    if kind in {"LGBM","LIGHTGBM"} and LGBMRegressor is not None:
        lr = float(ctx.get("lr", 1e-3))
        epochs = int(ctx.get("epochs", 20))
        return LGBMRegressor(
            n_estimators=max(50, epochs*5),
            learning_rate=min(0.2, max(1e-3, lr*10)),
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

    # 기본: 얕은 MLP
    return _mlp_fallback("MLP", ctx)


def get_classifier(kind: str, ctx: Dict[str, Any]):
    kind = (kind or "").upper()
    if kind in {"LGBM","LIGHTGBM"} and LGBMClassifier is not None:
        lr = float(ctx.get("lr", 1e-3))
        epochs = int(ctx.get("epochs", 20))
        return LGBMClassifier(
            n_estimators=max(50, epochs*5),
            learning_rate=min(0.2, max(1e-3, lr*10)),
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    # 기본: 로지스틱
    return LogisticRegression(max_iter=1000)
