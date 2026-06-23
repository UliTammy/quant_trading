"""
量化交易策略回测系统
支持：A股、ETF、开放式基金 — 双均线交叉、MACD、RSI、布林带、动量策略
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)


# ═══════════════════════════════════════════════════════
#  数据获取
# ═══════════════════════════════════════════════════════

def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取 A股 历史行情 (akshare)"""
    import akshare as ak
    raw = symbol.strip().lower()
    if raw.startswith(("sh", "sz")):
        code = raw[2:]
    elif raw.isdigit():
        code = raw
    else:
        raise ValueError(f"无法解析股票代码: {symbol}")

    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start.replace("-", ""),
                            end_date=end.replace("-", ""),
                            adjust="qfq")
    return _normalize_ohlcv(df, symbol)


def fetch_etf_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取 ETF 历史行情 — 优先 sina 源，回退 em"""
    import akshare as ak
    code = _clean_code(symbol)
    # sina 数据源（更稳定）
    try:
        mkt = "sh" if code.startswith("5") else "sz"
        df = ak.fund_etf_hist_sina(symbol=f"{mkt}{code}")
        if df is None or df.empty:
            raise ValueError("sina 返回空")
        # sina 的列名是英文，先统一
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        return _normalize_ohlcv(df, symbol)
    except Exception:
        pass

    # 回退 em
    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                             start_date=start.replace("-", ""),
                             end_date=end.replace("-", ""),
                             adjust="qfq")
    return _normalize_ohlcv(df, symbol)


def fetch_fund_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取开放式基金净值数据 (akshare)，转换为 OHLCV 近似格式"""
    import akshare as ak
    code = _clean_code(symbol)

    # 开放式基金只有净值，没有 OHLC
    nav_df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    if nav_df is None or nav_df.empty:
        # 回退：用 fund_etf_hist_em 尝试（部分 LOF 基金也可用）
        try:
            return fetch_etf_data(symbol, start, end)
        except Exception:
            raise ValueError(f"无法获取基金 {symbol} 的数据")

    nav_df = nav_df.rename(columns={"净值日期": "date", "单位净值": "nav"})
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date")
    nav_df = nav_df[(nav_df["date"] >= start) & (nav_df["date"] <= end)]

    if nav_df.empty:
        raise ValueError(f"基金 {symbol} 在指定日期范围内无数据")

    # 基金净值没有 OHLC，用当日净值的微小抖动模拟，确保策略可运行
    nav_df["close"] = nav_df["nav"]
    nav_df["open"] = nav_df["nav"] * (1 + np.random.uniform(-0.002, 0.002, len(nav_df)))
    nav_df["high"] = nav_df["nav"] * (1 + np.abs(np.random.normal(0, 0.005, len(nav_df))))
    nav_df["low"] = nav_df["nav"] * (1 - np.abs(np.random.normal(0, 0.005, len(nav_df))))
    nav_df["volume"] = np.random.randint(10000, 500000, len(nav_df))

    nav_df = nav_df.set_index("date")
    return nav_df


def _clean_code(symbol: str) -> str:
    """提取纯数字代码"""
    raw = symbol.strip().lower()
    for prefix in ("sh", "sz", "of", "bj"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _normalize_sina_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """sina 源的列名已经是英文 (date/open/high/low/close/volume)，只需裁剪"""
    if df is None or df.empty:
        raise ValueError(f"未获取到 {symbol} 的数据")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    for col in ["open", "close", "high", "low", "volume"]:
        if col not in df.columns:
            df[col] = df.get("close", df.iloc[:, -1])
    return df


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """统一 OHLCV 列名并返回标准 DataFrame"""
    if df is None or df.empty:
        raise ValueError(f"未获取到 {symbol} 的数据")

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" not in df.columns:
        raise ValueError(f"数据缺少日期列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # 确保必需列存在
    for col in ["open", "close", "high", "low", "volume"]:
        if col not in df.columns:
            df[col] = df.get("close", df.iloc[:, -1])

    return df


# 降级：模拟数据
def _generate_demo_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    if n < 50:
        raise ValueError("日期范围至少需要50个交易日")
    np.random.seed(hash(symbol) % 2**31)
    returns = np.random.normal(0.0003, 0.018, n)
    close = 10 + np.cumsum(returns) * 10
    close = np.maximum(close, 1)
    open_p = close * (1 + np.random.normal(0, 0.005, n))
    high = np.maximum(open_p, close) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = np.minimum(open_p, close) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    volume = np.random.randint(100000, 10000000, n)
    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def fetch_asset_data(asset_type: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """统一入口：根据资产类型路由到对应数据源"""
    try:
        if asset_type == "stock":
            return fetch_stock_data(symbol, start, end)
        elif asset_type == "etf":
            return fetch_etf_data(symbol, start, end)
        elif asset_type == "fund":
            return fetch_fund_data(symbol, start, end)
        else:
            raise ValueError(f"不支持的资产类型: {asset_type}")
    except ImportError:
        return _generate_demo_data(symbol, start, end)
    except Exception as e:
        if "akshare" in str(e) or "import" in str(e).lower():
            return _generate_demo_data(symbol, start, end)
        raise


# ═══════════════════════════════════════════════════════
#  技术指标（不变）
# ═══════════════════════════════════════════════════════

def calc_sma(series, window):
    return series.rolling(window=window).mean()

def calc_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(series, period=20, std=2.0):
    sma = calc_sma(series, period)
    std_dev = series.rolling(window=period).std()
    return sma + std * std_dev, sma, sma - std * std_dev

def calc_momentum(series, period=20):
    return series.diff(period) / series.shift(period)


# ═══════════════════════════════════════════════════════
#  策略引擎（不变）
# ═══════════════════════════════════════════════════════

def generate_signals(df, strategy, params):
    df = df.copy()
    close = df["close"]

    if strategy == "ma_crossover":
        s, l = params.get("short", 5), params.get("long", 20)
        df["sma_short"], df["sma_long"] = calc_sma(close, s), calc_sma(close, l)
        df["signal"] = 0
        df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
        df.loc[df["sma_short"] <= df["sma_long"], "signal"] = -1

    elif strategy == "macd":
        f, sl, sg = params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
        df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(close, f, sl, sg)
        df["signal"] = 0
        df.loc[df["macd"] > df["macd_signal"], "signal"] = 1
        df.loc[df["macd"] <= df["macd_signal"], "signal"] = -1

    elif strategy == "rsi":
        p, os, ob = params.get("period", 14), params.get("oversold", 30), params.get("overbought", 70)
        df["rsi"] = calc_rsi(close, p)
        df["signal"] = 0
        df.loc[df["rsi"] < os, "signal"] = 1
        df.loc[df["rsi"] > ob, "signal"] = -1

    elif strategy == "bollinger":
        p, s = params.get("period", 20), params.get("std", 2.0)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = calc_bollinger(close, p, s)
        df["signal"] = 0
        df.loc[close <= df["bb_lower"], "signal"] = 1
        df.loc[close >= df["bb_upper"], "signal"] = -1

    elif strategy == "momentum":
        p, th = params.get("period", 20), params.get("threshold", 0.02)
        df["momentum"] = calc_momentum(close, p)
        df["signal"] = 0
        df.loc[df["momentum"] > th, "signal"] = 1
        df.loc[df["momentum"] < -th, "signal"] = -1
    else:
        raise ValueError(f"未知策略: {strategy}")

    df["position_change"] = df["signal"].diff()
    df["buy"] = df["position_change"] == 2
    df["sell"] = df["position_change"] == -2
    if df["signal"].iloc[0] == 1:
        df.loc[df.index[0], "buy"] = True
    first_valid = df[df["signal"] != 0].index.min()
    if pd.notna(first_valid):
        df.loc[df.index < first_valid, ["buy", "sell"]] = False
    return df


# ═══════════════════════════════════════════════════════
#  回测引擎（不变）
# ═══════════════════════════════════════════════════════

def run_backtest(df, initial_capital=100000, commission=0.0003):
    df = df.copy().reset_index(drop=False)
    date_col = "date" if "date" in df.columns else "index"
    capital, shares = initial_capital, 0
    trades, equity_curve = [], []

    for _, row in df.iterrows():
        price, date_val = row["close"], row[date_col]

        if row.get("buy") and shares == 0 and capital > 0:
            shares = int(capital * (1 - commission) / price)
            cost = shares * price * (1 + commission)
            capital -= cost
            trades.append({"date": str(date_val)[:10], "type": "BUY", "price": round(price, 2),
                           "shares": shares, "amount": round(cost, 2)})
        elif row.get("sell") and shares > 0:
            revenue = shares * price * (1 - commission)
            capital += revenue
            trades.append({"date": str(date_val)[:10], "type": "SELL", "price": round(price, 2),
                           "shares": shares, "amount": round(revenue, 2),
                           "profit": round(revenue, 2)})
            shares = 0

        equity_curve.append({"date": str(date_val)[:10], "value": round(capital + shares * price, 2),
                             "price": round(price, 2)})

    if shares > 0:
        last_price = df.iloc[-1]["close"]
        capital += shares * last_price * (1 - commission)
        trades.append({"date": str(df.iloc[-1][date_col])[:10], "type": "SELL (强制清仓)",
                       "price": round(last_price, 2), "shares": shares,
                       "amount": round(shares * last_price * (1 - commission), 2)})

    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital
    eq = pd.DataFrame(equity_curve)
    eq["value"] = eq["value"].astype(float)
    eq["return"] = eq["value"].pct_change()
    days = len(eq)

    annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    eq["cummax"] = eq["value"].cummax()
    eq["drawdown"] = (eq["value"] - eq["cummax"]) / eq["cummax"]
    max_drawdown = eq["drawdown"].min()
    rf_daily = 0.03 / 252
    excess = eq["return"] - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    sell_trades = [t for t in trades if "SELL" in t["type"]]
    buy_trades = [t for t in trades if "BUY" in t["type"]]
    win_count = sum(1 for st in sell_trades for bt in reversed(buy_trades)
                    if bt["date"] <= st["date"] and st.get("amount", 0) > bt.get("amount", 0))
    win_rate = win_count / len(sell_trades) if sell_trades else 0

    buy_hold_return = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]

    return {
        "metrics": {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate": round(win_rate * 100, 1),
            "total_trades": len(sell_trades),
            "buy_hold_return": round(buy_hold_return * 100, 2),
        },
        "trades": trades,
        "equity_curve": equity_curve,
        "buy_signals": [{"date": str(row[date_col])[:10], "price": round(row["close"], 2)}
                        for _, row in df.iterrows() if row.get("buy")],
        "sell_signals": [{"date": str(row[date_col])[:10], "price": round(row["close"], 2)}
                         for _, row in df.iterrows() if row.get("sell")],
    }


# ═══════════════════════════════════════════════════════
#  资产清单
# ═══════════════════════════════════════════════════════

ASSETS = [
    # ── A 股 ──
    {"symbol": "600519", "name": "贵州茅台",       "type": "stock", "market": "SH"},
    {"symbol": "000858", "name": "五粮液",         "type": "stock", "market": "SZ"},
    {"symbol": "601318", "name": "中国平安",       "type": "stock", "market": "SH"},
    {"symbol": "000333", "name": "美的集团",       "type": "stock", "market": "SZ"},
    {"symbol": "600036", "name": "招商银行",       "type": "stock", "market": "SH"},
    {"symbol": "000651", "name": "格力电器",       "type": "stock", "market": "SZ"},
    {"symbol": "002415", "name": "海康威视",       "type": "stock", "market": "SZ"},
    {"symbol": "600276", "name": "恒瑞医药",       "type": "stock", "market": "SH"},
    {"symbol": "000725", "name": "京东方A",        "type": "stock", "market": "SZ"},
    {"symbol": "601012", "name": "隆基绿能",       "type": "stock", "market": "SH"},
    {"symbol": "300750", "name": "宁德时代",       "type": "stock", "market": "SZ"},
    {"symbol": "002594", "name": "比亚迪",         "type": "stock", "market": "SZ"},
    {"symbol": "600900", "name": "长江电力",       "type": "stock", "market": "SH"},
    {"symbol": "000001", "name": "平安银行",       "type": "stock", "market": "SZ"},
    {"symbol": "601398", "name": "工商银行",       "type": "stock", "market": "SH"},
    {"symbol": "600030", "name": "中信证券",       "type": "stock", "market": "SH"},
    {"symbol": "000002", "name": "万科A",          "type": "stock", "market": "SZ"},
    {"symbol": "601857", "name": "中国石油",       "type": "stock", "market": "SH"},
    {"symbol": "300059", "name": "东方财富",       "type": "stock", "market": "SZ"},
    {"symbol": "688981", "name": "中芯国际",       "type": "stock", "market": "SH"},

    # ── ETF ──
    {"symbol": "510050", "name": "上证50ETF",      "type": "etf",  "market": "SH"},
    {"symbol": "510300", "name": "沪深300ETF",     "type": "etf",  "market": "SH"},
    {"symbol": "510500", "name": "中证500ETF",     "type": "etf",  "market": "SH"},
    {"symbol": "159915", "name": "创业板ETF",      "type": "etf",  "market": "SZ"},
    {"symbol": "588000", "name": "科创50ETF",      "type": "etf",  "market": "SH"},
    {"symbol": "512880", "name": "证券ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "512010", "name": "医药ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "512100", "name": "中证1000ETF",    "type": "etf",  "market": "SH"},
    {"symbol": "159949", "name": "创业板50ETF",    "type": "etf",  "market": "SZ"},
    {"symbol": "513100", "name": "纳指ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "159920", "name": "恒生ETF",        "type": "etf",  "market": "SZ"},
    {"symbol": "518880", "name": "黄金ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "511260", "name": "十年国债ETF",    "type": "etf",  "market": "SH"},
    {"symbol": "512660", "name": "军工ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "515790", "name": "光伏ETF",        "type": "etf",  "market": "SH"},
    {"symbol": "516510", "name": "云计算ETF",      "type": "etf",  "market": "SH"},
    {"symbol": "562500", "name": "机器人ETF",      "type": "etf",  "market": "SH"},
    {"symbol": "513050", "name": "中概互联ETF",    "type": "etf",  "market": "SH"},

    # ── 开放式基金 ──
    {"symbol": "110022", "name": "易方达消费行业", "type": "fund", "market": "OF"},
    {"symbol": "001632", "name": "天弘中证食品饮料","type": "fund", "market": "OF"},
    {"symbol": "005827", "name": "易方达蓝筹精选", "type": "fund", "market": "OF"},
    {"symbol": "161725", "name": "招商中证白酒",   "type": "fund", "market": "OF"},
    {"symbol": "320007", "name": "诺安成长混合",   "type": "fund", "market": "OF"},
    {"symbol": "002939", "name": "广发创新升级",   "type": "fund", "market": "OF"},
    {"symbol": "501057", "name": "汇添富新能源车", "type": "fund", "market": "OF"},
    {"symbol": "000913", "name": "农银医疗保健",   "type": "fund", "market": "OF"},
    {"symbol": "260108", "name": "景顺长城新兴成长","type": "fund", "market": "OF"},
    {"symbol": "003095", "name": "中欧医疗健康",   "type": "fund", "market": "OF"},
    {"symbol": "001475", "name": "易方达国防军工", "type": "fund", "market": "OF"},
    {"symbol": "004997", "name": "广发高端制造",   "type": "fund", "market": "OF"},
]

TYPE_LABELS = {"stock": "A股", "etf": "ETF", "fund": "基金"}


# ═══════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/strategies")
def api_strategies():
    return jsonify({"strategies": [
        {"id": "ma_crossover", "name": "双均线交叉", "desc": "短期均线上穿长期均线买入，下穿卖出",
         "params": [
             {"key": "short", "name": "短期均线", "type": "int", "default": 5, "min": 2, "max": 60},
             {"key": "long",  "name": "长期均线", "type": "int", "default": 20, "min": 5, "max": 200},
         ]},
        {"id": "macd", "name": "MACD 策略", "desc": "MACD线金叉买入，死叉卖出",
         "params": [
             {"key": "fast",   "name": "快线周期", "type": "int", "default": 12, "min": 3, "max": 50},
             {"key": "slow",   "name": "慢线周期", "type": "int", "default": 26, "min": 5, "max": 100},
             {"key": "signal", "name": "信号线周期","type": "int", "default": 9, "min": 3, "max": 30},
         ]},
        {"id": "rsi", "name": "RSI 策略", "desc": "RSI超卖时买入，超买时卖出",
         "params": [
             {"key": "period",     "name": "RSI周期",  "type": "int", "default": 14, "min": 5, "max": 50},
             {"key": "oversold",   "name": "超卖阈值",  "type": "int", "default": 30, "min": 10, "max": 40},
             {"key": "overbought", "name": "超买阈值",  "type": "int", "default": 70, "min": 60, "max": 90},
         ]},
        {"id": "bollinger", "name": "布林带策略", "desc": "触及下轨买入，触及上轨卖出",
         "params": [
             {"key": "period", "name": "布林带周期", "type": "int",   "default": 20,  "min": 5,  "max": 100},
             {"key": "std",    "name": "标准差倍数", "type": "float", "default": 2.0, "min": 1.0,"max": 4.0,"step": 0.1},
         ]},
        {"id": "momentum", "name": "动量策略", "desc": "动量超过阈值做多，低于反向阈值做空",
         "params": [
             {"key": "period",    "name": "动量周期",   "type": "int",   "default": 20,   "min": 5,   "max": 60},
             {"key": "threshold", "name": "动量阈值(%)", "type": "float", "default": 0.02, "min": 0.005,"max": 0.2,"step": 0.005},
         ]},
    ]})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "600519")
        asset_type = data.get("asset_type", "stock")
        strategy = data.get("strategy", "ma_crossover")
        params = data.get("params", {})
        start = data.get("start", (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")).replace("-", "")
        end = data.get("end", datetime.now().strftime("%Y%m%d")).replace("-", "")
        capital = float(data.get("capital", 100000))

        df = fetch_asset_data(asset_type, symbol, start, end)
        df = generate_signals(df, strategy, params)
        result = run_backtest(df, initial_capital=capital)

        kline = []
        for idx, row in df.iterrows():
            ds = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            kline.append([ds, round(float(row["open"]), 2), round(float(row["close"]), 2),
                          round(float(row["low"]), 2), round(float(row["high"]), 2)])

        return jsonify({"success": True, "symbol": symbol, "asset_type": asset_type,
                        "strategy": strategy, "kline": kline, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/search_stock", methods=["GET"])
def api_search_stock():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    ql = q.lower()
    if ql.isdigit():
        results = [a for a in ASSETS if ql in a["symbol"]]
    else:
        results = [a for a in ASSETS
                   if ql in a["name"].lower() or ql in a["symbol"].lower()]

    return jsonify(results[:15])


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "600519")
        asset_type = data.get("asset_type", "stock")
        start = data.get("start", (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")).replace("-", "")
        end = data.get("end", datetime.now().strftime("%Y%m%d")).replace("-", "")
        capital = float(data.get("capital", 100000))

        df = fetch_asset_data(asset_type, symbol, start, end)

        strategies_to_compare = [
            {"id": "ma_crossover", "name": "双均线交叉", "params": {"short": 5, "long": 20}},
            {"id": "macd",         "name": "MACD",       "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"id": "rsi",          "name": "RSI",        "params": {"period": 14, "oversold": 30, "overbought": 70}},
            {"id": "bollinger",    "name": "布林带",     "params": {"period": 20, "std": 2.0}},
            {"id": "momentum",     "name": "动量策略",   "params": {"period": 20, "threshold": 0.02}},
        ]

        results, equity_data = [], {}
        buy_hold = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100

        for st in strategies_to_compare:
            try:
                sdf = generate_signals(df.copy(), st["id"], st["params"])
                r = run_backtest(sdf, initial_capital=capital)
                results.append({"name": st["name"], **r["metrics"]})
                equity_data[st["name"]] = r["equity_curve"]
            except Exception as e:
                results.append({"name": st["name"], "error": str(e)})

        return jsonify({"success": True, "symbol": symbol, "asset_type": asset_type,
                        "results": results, "equity_data": equity_data,
                        "buy_hold_return": round(buy_hold, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("Quant Trading System: http://127.0.0.1:5001")
    app.run(debug=False, host="0.0.0.0", port=5001)
