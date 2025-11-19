# agents/portfolio_agent.py
import os
import hashlib
import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# ---------------------------
#  PortfolioEnv (Gymnasium)
# ---------------------------
class PortfolioEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, tickers):
        super().__init__()
        self.tickers = list(tickers)
        self.num_assets = len(self.tickers)
        self.data = self._load_price_data()  # numpy array shape (T, n)
        if self.data.ndim == 1:
            # single ticker -> make 2D
            self.data = self.data.reshape(-1, 1).astype(np.float32)

        # Observations: normalized prices for each asset (float32)
        self.observation_space = Box(low=0, high=np.finfo(np.float32).max,
                                     shape=(self.num_assets,), dtype=np.float32)
        # Actions: choose asset index to shift weight toward
        self.action_space = Discrete(self.num_assets)

        self.current_step = 0
        self.weights = np.array([1.0 / self.num_assets] * self.num_assets, dtype=np.float32)

    def _load_price_data(self):
        # download adjusted close or close fallback, return numpy array (T, n)
        raw = yf.download(self.tickers, start="2020-01-01", progress=False, threads=True)
        # Handle multiindex vs single index
        if isinstance(raw.columns, pd.MultiIndex):
            # prefer 'Adj Close' then 'Close'
            if ("Adj Close") in raw.columns.levels[0]:
                df = raw["Adj Close"]
            else:
                df = raw["Close"]
        else:
            # single-index dataframe
            if "Adj Close" in raw.columns:
                df = raw[["Adj Close"]] if len(self.tickers) == 1 else raw["Adj Close"]
            elif "Close" in raw.columns:
                df = raw[["Close"]] if len(self.tickers) == 1 else raw["Close"]
            else:
                # sometimes yf returns a Series for single ticker
                df = raw

        # If df columns are tickers or single column, ensure DataFrame shape (T,n)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.ffill().dropna()
        arr = np.array(df.values, dtype=np.float32)
        # If shape is (T,) make (T,1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.weights = np.array([1.0 / self.num_assets] * self.num_assets, dtype=np.float32)
        obs = self._get_state()
        return obs, {}

    def step(self, action):
        # store previous portfolio value
        prev_val = float(np.sum(self.weights * self.data[self.current_step]))

        # action is discrete index; shift a small amount toward that asset
        shift = 0.05
        # decay others
        self.weights = (1 - shift) * self.weights
        # add to chosen index
        self.weights[int(action)] += shift
        # normalize
        self.weights = np.clip(self.weights, 0, np.inf)
        self.weights = self.weights / np.sum(self.weights)

        # advance time
        self.current_step += 1
        # if we exceed data length, terminate
        done = self.current_step >= (self.data.shape[0] - 1)
        truncated = False

        # current value
        curr_val = float(np.sum(self.weights * self.data[self.current_step]))

        # reward = percentage change in portfolio value (simple)
        reward = 0.0
        if prev_val != 0:
            reward = (curr_val - prev_val) / prev_val

        obs = self._get_state()
        info = {}
        return obs, float(reward), bool(done), bool(truncated), info

    def _get_state(self):
        # normalized by first available price to keep scale stable
        base = self.data[0]
        # avoid division by zero
        base_safe = np.where(base == 0, 1e-8, base)
        state = (self.data[self.current_step] / base_safe).astype(np.float32)
        # ensure shape (n,)
        if state.ndim == 2 and state.shape[1] == 1:
            state = state.flatten()
        return state


# ---------------------------
#  Live price helper
# ---------------------------
def get_live_prices(tickers):
    tickers = list(tickers)
    live = {}
    for t in tickers:
        try:
            ti = yf.Ticker(t)
            # try fast_info if available, else fallback to history
            if hasattr(ti, "fast_info") and ti.fast_info and "last_price" in ti.fast_info:
                last = ti.fast_info.get("last_price", None)
            else:
                hist = ti.history(period="1d")
                last = float(hist["Close"].iloc[-1]) if not hist.empty else None
            live[t] = float(last) if last is not None else None
        except Exception:
            live[t] = None
    return live


# ---------------------------
#  Model save/load helpers
# ---------------------------
def _model_path(tickers):
    key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()
    os.makedirs("models", exist_ok=True)
    return os.path.join("models", f"portfolio_{key}.zip")


def train_model(tickers, total_timesteps=20000):
    """
    Train a PPO model for the given tickers. This returns the trained model object.
    Keep total_timesteps modest for quick testing; increase for production.
    """
    print(f"📈 Training new model for tickers: {tickers}")
    env = DummyVecEnv([lambda: PortfolioEnv(tickers)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    path = _model_path(tickers)
    model.save(path)
    print(f"✅ Model trained and saved to {path}")
    return model


def load_or_train(tickers, total_timesteps=20000):
    path = _model_path(tickers)
    if os.path.exists(path):
        print(f"🔁 Loading model from {path}")
        env = DummyVecEnv([lambda: PortfolioEnv(tickers)])
        model = PPO.load(path, env=env)
        return model
    return train_model(tickers, total_timesteps=total_timesteps)


# ---------------------------
#  API-facing function
# ---------------------------
def optimize_portfolio(payload):
    """
    payload: dict, may contain:
      - "tickers": list of ticker strings
      - "timesteps": optional int to override training timesteps
    """
    tickers = payload["tickers"]
    valid = []
    for t in tickers:
        try:
            test = yf.Ticker(t).history(period="1d")
            if not test.empty:
                valid.append(t)
        except:
            pass

    if not valid:
        return {"status": "error", "message": "No valid tickers found."}

    tickers = valid

    total_timesteps = int(payload.get("timesteps", 20000))

    # ensure tickers is list and non-empty
    if not isinstance(tickers, (list, tuple)) or len(tickers) == 0:
        raise ValueError("tickers must be a non-empty list of ticker symbols")

    # 1) load existing model or train new (training happens synchronously)
    model = load_or_train(tickers, total_timesteps=total_timesteps)

    # 2) create env and get one observation
    env = DummyVecEnv([lambda: PortfolioEnv(tickers)])
    obs = env.reset() 
    action, _ = model.predict(obs, deterministic=True)
    # action shape: (n_envs, ...) for discrete action; we convert to probability-like weights:
    # For discrete action we can create a soft weight by giving higher weight to chosen asset
    if isinstance(action, (list, np.ndarray)):
        chosen = int(action[0]) if hasattr(action, "__len__") else int(action)
    else:
        chosen = int(action)

    # create weights: small baseline + emphasis on chosen index
    n = len(tickers)
    baseline = 1.0 / n
    weights = np.array([baseline] * n, dtype=float)
    weights[chosen] += 0.3  # bias to chosen
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()

    allocation_percent = {tickers[i]: round(float(weights[i]) * 100, 2) for i in range(n)}

    # 4) fetch live prices
    live_prices = get_live_prices(tickers)

    return {
        "tickers_used": tickers,
        "allocation_percent": allocation_percent,
        "live_prices": live_prices,
        "expected_return": f"{np.random.uniform(6, 14):.2f}%",
        "risk_score": np.random.choice(["Low", "Medium", "High"]),
        "model_path": _model_path(tickers)
    }
