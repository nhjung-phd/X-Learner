# core/models_dl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def infer_window_and_series(X: pd.DataFrame) -> Tuple[int, int, Dict[str, list]]:
    """
    build_lag_features()가 만든 칼럼 패턴: {name}_lag{k}
    예: ret_T1_lag1 ... ret_T1_lagW, ret_T0_lag1 ... ret_T0_lagW
    반환: (window=W, n_series=S, groups={name: [cols...]})
    """
    groups: Dict[str, list] = {}
    for c in X.columns:
        if "_lag" in c:
            base = c.split("_lag")[0]
        else:
            base = c
        groups.setdefault(base, []).append(c)
    # 각 시리즈 내에서 lag1..W 정렬
    for k, cols in groups.items():
        groups[k] = sorted(cols, key=lambda x: int(x.split("_lag")[-1]) if "_lag" in x else 0)
    # window는 한 그룹 칼럼 수로 추정
    window = len(next(iter(groups.values())))
    n_series = len(groups)
    return window, n_series, groups


def to_3d_sequence(X: pd.DataFrame, window: int) -> np.ndarray:
    """
    (N, W*S) 형태의 lagged feature를 (N, W, S) 3D 텐서로 변환.
    그룹 순서는 컬럼 순서 기반으로 결정하되, 시간 순서는 과거→현재로 맞춤.
    """
    W, S, groups = infer_window_and_series(X)
    # 사용자가 window를 바꿨다면, 실제 칼럼 기반 W를 신뢰
    W = W
    series_names = list(groups.keys())
    # (N, W, S)
    N = X.shape[0]
    X3 = np.zeros((N, W, S), dtype=np.float32)
    for si, name in enumerate(series_names):
        cols = groups[name]
        # cols는 lag1..lagW 오름차순이므로, 시간축은 과거→현재로 이미 맞춰져 있음
        X3[:, :, si] = X[cols].values.astype(np.float32)
    return X3


class KerasSeqRegressor:
    """
    간단한 시퀀스 회귀 래퍼:
      - kind: 'LSTM'|'GRU'|'RNN'|'CNN'
      - fit(X,y): X는 lagged feature DataFrame, y는 pd.Series
      - predict(X): 1D numpy 반환
    """
    def __init__(self, kind: str, ctx: Dict[str, Any]):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        self.kind = (kind or "LSTM").upper()
        self.lr = float(ctx.get("lr", 1e-3))
        self.epochs = int(ctx.get("epochs", 20))
        self.batch = int(ctx.get("batch", 64))
        self.early_stopping = bool(ctx.get("early_stopping", True))
        self.model: Optional[keras.Model] = None

    def _build(self, input_shape: Tuple[int,int]) -> keras.Model:
        W, S = input_shape  # (timesteps, features)
        inputs = keras.Input(shape=(W, S))
        if self.kind == "LSTM":
            x = keras.layers.LSTM(128, return_sequences=False)(inputs)
        elif self.kind == "GRU":
            x = keras.layers.GRU(128, return_sequences=False)(inputs)
        elif self.kind == "RNN":
            x = keras.layers.SimpleRNN(128, return_sequences=False)(inputs)
        elif self.kind == "CNN":
            x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
            x = keras.layers.GlobalAveragePooling1D()(x)
        else:
            x = keras.layers.LSTM(64, return_sequences=False)(inputs)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss="mse")
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X3 = to_3d_sequence(X, window=0)  # window는 infer에서 결정
        yv = y.loc[X.index].values.astype(np.float32).reshape(-1,1)
        self.model = self._build((X3.shape[1], X3.shape[2]))
        cb = []
        if self.early_stopping:
            cb.append(keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True))
        self.model.fit(
            X3, yv,
            epochs=self.epochs, batch_size=self.batch,
            verbose=0, callbacks=cb
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not trained"
        X3 = to_3d_sequence(X, window=0)
        pred = self.model.predict(X3, verbose=0).reshape(-1)
        return pred
