"""
量化交易策略回测系统
支持多种策略：双均线交叉、MACD、RSI、布林带、动量策略
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# ─── 数据获取 ───────────────────────────────────────────
def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取股票历史行情数据"""
    try:
        import akshare as ak

        # 清理 symbol（如 sh600000 / sz000001）
        raw = symbol.strip().lower()
        if raw.startswith("sh") or raw.startswith("sz"):
            code = raw[2:]
            mkt = raw[:2]
        elif raw.isdigit():
            code = raw
            mkt = "sh" if code.startswith(("6", "9")) else "sz"
        else:
            raise ValueError(f"无法解析股票代码: {symbol}")

        full_symbol = f"{mkt}{code}"
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start.replace("-", ""),
                                end_date=end.replace("-", ""),
                                adjust="qfq")
        if df.empty:
            raise ValueError("未获取到数据")

        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        df["symbol"] = full_symbol
        return df

    except ImportError:
        # 生成模拟数据用于演示
        return _generate_demo_data(symbol, start, end)
    except Exception as e:
        print(f"数据获取失败: {e}, 使用模拟数据")
        return _generate_demo_data(symbol, start, end)


def _generate_demo_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """生成模拟K线数据"""
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

    return pd.DataFrame({
        "open": open_p, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


# ─── 技术指标 ───────────────────────────────────────────
def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def calc_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma = calc_sma(series, period)
    std_dev = series.rolling(window=period).std()
    return sma + std * std_dev, sma, sma - std * std_dev

def calc_momentum(series: pd.Series, period: int = 20) -> pd.Series:
    return series.diff(period) / series.shift(period)


# ─── 策略引擎 ───────────────────────────────────────────
def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """根据策略生成买卖信号"""
    df = df.copy()
    close = df["close"]

    if strategy == "ma_crossover":
        short = params.get("short", 5)
        long = params.get("long", 20)
        df["sma_short"] = calc_sma(close, short)
        df["sma_long"] = calc_sma(close, long)
        df["signal"] = 0
        df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
        df.loc[df["sma_short"] <= df["sma_long"], "signal"] = -1

    elif strategy == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        sig = params.get("signal", 9)
        df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(close, fast, slow, sig)
        df["signal"] = 0
        df.loc[df["macd"] > df["macd_signal"], "signal"] = 1
        df.loc[df["macd"] <= df["macd_signal"], "signal"] = -1

    elif strategy == "rsi":
        period = params.get("period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        df["rsi"] = calc_rsi(close, period)
        df["signal"] = 0
        df.loc[df["rsi"] < oversold, "signal"] = 1
        df.loc[df["rsi"] > overbought, "signal"] = -1

    elif strategy == "bollinger":
        period = params.get("period", 20)
        std = params.get("std", 2.0)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = calc_bollinger(close, period, std)
        df["signal"] = 0
        df.loc[close <= df["bb_lower"], "signal"] = 1
        df.loc[close >= df["bb_upper"], "signal"] = -1

    elif strategy == "momentum":
        period = params.get("period", 20)
        threshold = params.get("threshold", 0.02)
        df["momentum"] = calc_momentum(close, period)
        df["signal"] = 0
        df.loc[df["momentum"] > threshold, "signal"] = 1
        df.loc[df["momentum"] < -threshold, "signal"] = -1

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 生成交易信号（信号变化时触发）
    df["position_change"] = df["signal"].diff()
    df["buy"] = df["position_change"] == 2       # -1 → 1
    df["sell"] = df["position_change"] == -2     # 1 → -1
    # 初始建仓
    if df["signal"].iloc[0] == 1:
        df.loc[df.index[0], "buy"] = True
    # 滤除 NaN 期间
    first_valid = df[df["signal"] != 0].index.min()
    if pd.notna(first_valid):
        df.loc[df.index < first_valid, ["buy", "sell"]] = False

    return df


# ─── 回测引擎 ───────────────────────────────────────────
def run_backtest(df: pd.DataFrame, initial_capital: float = 100000,
                 commission: float = 0.0003) -> dict:
    """执行回测并返回结果"""
    df = df.copy()
    df = df.reset_index(drop=False)
    date_col = "date" if "date" in df.columns else "index"

    capital = initial_capital
    shares = 0
    trades = []
    equity_curve = []

    for i, row in df.iterrows():
        price = row["close"]
        date_val = row[date_col]

        if row.get("buy") and shares == 0 and capital > 0:
            shares = int(capital * (1 - commission) / price)
            cost = shares * price * (1 + commission)
            capital -= cost
            trades.append({
                "date": str(date_val)[:10],
                "type": "BUY",
                "price": round(price, 2),
                "shares": shares,
                "amount": round(cost, 2),
            })

        elif row.get("sell") and shares > 0:
            revenue = shares * price * (1 - commission)
            profit = revenue - sum(
                t["amount"] for t in trades if t["type"] == "BUY"
            )
            trades.append({
                "date": str(date_val)[:10],
                "type": "SELL",
                "price": round(price, 2),
                "shares": shares,
                "amount": round(revenue, 2),
                "profit": round(revenue, 2),
            })
            capital += revenue
            shares = 0

        total_value = capital + shares * price
        equity_curve.append({
            "date": str(date_val)[:10],
            "value": round(total_value, 2),
            "price": round(price, 2),
        })

    # 最终清仓
    if shares > 0:
        last_price = df.iloc[-1]["close"]
        capital += shares * last_price * (1 - commission)
        trades.append({
            "date": str(df.iloc[-1][date_col])[:10],
            "type": "SELL (强制清仓)",
            "price": round(last_price, 2),
            "shares": shares,
            "amount": round(shares * last_price * (1 - commission), 2),
        })

    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital

    # 计算绩效指标
    eq = pd.DataFrame(equity_curve)
    eq["value"] = eq["value"].astype(float)
    eq["return"] = eq["value"].pct_change()

    # 年化收益率
    days = len(eq)
    annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

    # 最大回撤
    eq["cummax"] = eq["value"].cummax()
    eq["drawdown"] = (eq["value"] - eq["cummax"]) / eq["cummax"]
    max_drawdown = eq["drawdown"].min()

    # 夏普比率（简化）
    rf_daily = 0.03 / 252
    excess = eq["return"] - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # 胜率
    buy_trades = [t for t in trades if t["type"].startswith("BUY")]
    sell_trades = [t for t in trades if t["type"].startswith("SELL")]
    win_count = 0
    for st in sell_trades:
        for bt in reversed(buy_trades):
            if bt["date"] <= st["date"]:
                if st.get("amount", 0) > bt.get("amount", 0):
                    win_count += 1
                break
    win_rate = win_count / len(sell_trades) if sell_trades else 0

    # 买入持有基准
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
        "buy_signals": [
            {"date": str(row[date_col])[:10], "price": round(row["close"], 2)}
            for _, row in df.iterrows() if row.get("buy")
        ],
        "sell_signals": [
            {"date": str(row[date_col])[:10], "price": round(row["close"], 2)}
            for _, row in df.iterrows() if row.get("sell")
        ],
    }


# ─── API 路由 ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/strategies")
def api_strategies():
    """返回可用策略列表"""
    return jsonify({
        "strategies": [
            {
                "id": "ma_crossover",
                "name": "双均线交叉",
                "desc": "短期均线上穿长期均线买入，下穿卖出",
                "params": [
                    {"key": "short", "name": "短期均线", "type": "int", "default": 5, "min": 2, "max": 60},
                    {"key": "long", "name": "长期均线", "type": "int", "default": 20, "min": 5, "max": 200},
                ]
            },
            {
                "id": "macd",
                "name": "MACD 策略",
                "desc": "MACD线金叉买入，死叉卖出",
                "params": [
                    {"key": "fast", "name": "快线周期", "type": "int", "default": 12, "min": 3, "max": 50},
                    {"key": "slow", "name": "慢线周期", "type": "int", "default": 26, "min": 5, "max": 100},
                    {"key": "signal", "name": "信号线周期", "type": "int", "default": 9, "min": 3, "max": 30},
                ]
            },
            {
                "id": "rsi",
                "name": "RSI 策略",
                "desc": "RSI超卖时买入，超买时卖出",
                "params": [
                    {"key": "period", "name": "RSI周期", "type": "int", "default": 14, "min": 5, "max": 50},
                    {"key": "oversold", "name": "超卖阈值", "type": "int", "default": 30, "min": 10, "max": 40},
                    {"key": "overbought", "name": "超买阈值", "type": "int", "default": 70, "min": 60, "max": 90},
                ]
            },
            {
                "id": "bollinger",
                "name": "布林带策略",
                "desc": "触及下轨买入，触及上轨卖出",
                "params": [
                    {"key": "period", "name": "布林带周期", "type": "int", "default": 20, "min": 5, "max": 100},
                    {"key": "std", "name": "标准差倍数", "type": "float", "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1},
                ]
            },
            {
                "id": "momentum",
                "name": "动量策略",
                "desc": "动量超过阈值做多，低于反向阈值做空",
                "params": [
                    {"key": "period", "name": "动量周期", "type": "int", "default": 20, "min": 5, "max": 60},
                    {"key": "threshold", "name": "动量阈值(%)", "type": "float", "default": 0.02, "min": 0.005, "max": 0.2, "step": 0.005},
                ]
            },
        ]
    })


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """执行回测"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "600519")
        strategy = data.get("strategy", "ma_crossover")
        params = data.get("params", {})
        start = data.get("start", (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d"))
        end = data.get("end", datetime.now().strftime("%Y%m%d"))
        capital = float(data.get("capital", 100000))

        # 格式标准化
        start = start.replace("-", "")
        end = end.replace("-", "")

        df = fetch_stock_data(symbol, start, end)
        df = generate_signals(df, strategy, params)
        result = run_backtest(df, initial_capital=capital)

        # 构建 K 线数据（前端画图用）
        kline = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            kline.append([
                date_str,
                round(float(row["open"]), 2),
                round(float(row["close"]), 2),
                round(float(row["low"]), 2),
                round(float(row["high"]), 2),
            ])

        return jsonify({
            "success": True,
            "symbol": symbol,
            "strategy": strategy,
            "kline": kline,
            **result
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/search_stock", methods=["GET"])
def api_search_stock():
    """搜索股票"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    # 常用股票列表
    stocks = [
        {"symbol": "600519", "name": "贵州茅台", "market": "SH"},
        {"symbol": "000858", "name": "五粮液", "market": "SZ"},
        {"symbol": "601318", "name": "中国平安", "market": "SH"},
        {"symbol": "000333", "name": "美的集团", "market": "SZ"},
        {"symbol": "600036", "name": "招商银行", "market": "SH"},
        {"symbol": "000651", "name": "格力电器", "market": "SZ"},
        {"symbol": "002415", "name": "海康威视", "market": "SZ"},
        {"symbol": "600276", "name": "恒瑞医药", "market": "SH"},
        {"symbol": "000725", "name": "京东方A", "market": "SZ"},
        {"symbol": "601012", "name": "隆基绿能", "market": "SH"},
        {"symbol": "300750", "name": "宁德时代", "market": "SZ"},
        {"symbol": "002594", "name": "比亚迪", "market": "SZ"},
        {"symbol": "600900", "name": "长江电力", "market": "SH"},
        {"symbol": "000001", "name": "平安银行", "market": "SZ"},
        {"symbol": "601398", "name": "工商银行", "market": "SH"},
        {"symbol": "600030", "name": "中信证券", "market": "SH"},
        {"symbol": "000002", "name": "万科A", "market": "SZ"},
        {"symbol": "601857", "name": "中国石油", "market": "SH"},
        {"symbol": "300059", "name": "东方财富", "market": "SZ"},
        {"symbol": "688981", "name": "中芯国际", "market": "SH"},
    ]

    if q.isdigit():
        results = [s for s in stocks if q in s["symbol"]]
    else:
        results = [s for s in stocks if q.lower() in s["name"].lower() or q.lower() in s["symbol"].lower()]

    return jsonify(results[:10])


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """多策略对比"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "600519")
        start = data.get("start", (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d"))
        end = data.get("end", datetime.now().strftime("%Y%m%d"))
        capital = float(data.get("capital", 100000))

        start = start.replace("-", "")
        end = end.replace("-", "")

        df = fetch_stock_data(symbol, start, end)

        strategies_to_compare = [
            {"id": "ma_crossover", "name": "双均线交叉", "params": {"short": 5, "long": 20}},
            {"id": "macd", "name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"id": "rsi", "name": "RSI", "params": {"period": 14, "oversold": 30, "overbought": 70}},
            {"id": "bollinger", "name": "布林带", "params": {"period": 20, "std": 2.0}},
            {"id": "momentum", "name": "动量策略", "params": {"period": 20, "threshold": 0.02}},
        ]

        results = []
        equity_data = {}

        # 买入持有基准
        buy_hold = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100

        for st in strategies_to_compare:
            try:
                sdf = generate_signals(df.copy(), st["id"], st["params"])
                r = run_backtest(sdf, initial_capital=capital)
                results.append({
                    "name": st["name"],
                    **r["metrics"],
                })
                equity_data[st["name"]] = r["equity_curve"]
            except Exception as e:
                results.append({"name": st["name"], "error": str(e)})

        return jsonify({
            "success": True,
            "symbol": symbol,
            "results": results,
            "equity_data": equity_data,
            "buy_hold_return": round(buy_hold, 2),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("Quant Trading System: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
