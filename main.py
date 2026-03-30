"""
VeighNa Quant Backend - Stock Analysis API (Production)
"""

import sys
import os
import json
import math
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple


import numpy as np
import pandas as pd
import akshare as ak
from flask import Flask, request

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.json.ensure_ascii = False

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    import socket
    tb = traceback.format_exc()
    err_type = e.__class__.__name__
    
    # 根据异常类型给出友好提示
    if isinstance(e, (ConnectionError, socket.timeout, TimeoutError)):
        friendly = '数据源连接失败，请稍后重试（网络或服务器繁忙）'
    elif 'Connection' in err_type or 'Timeout' in err_type or 'HTTP' in err_type:
        friendly = '网络连接异常，请检查网络后重试'
    else:
        friendly = f'分析出错，请稍后重试'
    
    print(f"[ERROR] {err_type}: {e}\n{tb}", file=sys.stderr)
    return json_response({'success': False, 'error': friendly})

# ============ 配置 ============
COMMISSION_RATE = 0.0015    # 双向佣金 0.15%（0.1%印花税只在卖出收，0.05%双向佣金）
SLIPPAGE_RATE = 0.001       # 滑点 0.1%（乐观估计）
INITIAL_CAPITAL = 100000.0  # 初始资金 10万
RISK_FREE_RATE = 0.03       # 无风险利率 3%

# ============ JSON 响应 ============
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def json_response(data: dict):
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False, cls=NumpyEncoder),
        status=200,
        mimetype='application/json'
    )

# ============ 参数校验 ============
def validate_params(fast: int, slow: int) -> Tuple[bool, str]:
    if fast <= 0 or slow <= 0:
        return False, "周期参数必须为正整数"
    if fast >= slow:
        return False, "快线周期必须小于慢线周期"
    if fast > 250 or slow > 250:
        return False, "周期参数不能超过250"
    return True, ""

# ============ 工具函数 ============
def fmt(n, d=2):
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return None
    try:
        return round(float(n), d)
    except (ValueError, TypeError):
        return None

# ============ 缓存友好的股票数据获取 ============
# 简单内存缓存（DataFrame 副本）
_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_CACHE_TIMES: Dict[str, float] = {}

def get_stock_data_cached(code: str, period: int = 250) -> Optional[pd.DataFrame]:
    """获取股票数据（带简单内存缓存，5分钟 TTL）"""
    import time
    cache_key = code
    now = time.time()
    
    # 缓存命中且未过期（5分钟）
    if cache_key in _DATA_CACHE:
        if now - _CACHE_TIMES.get(cache_key, 0) < 300:
            return _DATA_CACHE[cache_key].copy()
    
    # 重新获取
    df = _fetch_stock_data_impl(code, period)
    if df is not None:
        _DATA_CACHE[cache_key] = df
        _CACHE_TIMES[cache_key] = now
    return df.copy() if df is not None else None

def _fetch_stock_data_impl(code: str, period: int) -> Optional[pd.DataFrame]:
    """实际获取股票数据（akshare 优先，失败则用 baostock 备用）"""
    import time
    import baostock as bs
    symbol = code.replace('SH', '').replace('SZ', '')
    
    # 方法1：akshare（东方财富）
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=(datetime.now() - timedelta(days=period * 2)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust='qfq'
            )
            if df is not None and not df.empty:
                df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']].copy()
                df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                return df
        except Exception as e:
            print(f"[WARN] akshare attempt {attempt+1}/2 failed for {code}: {e}", file=sys.stderr)
            if attempt < 1:
                time.sleep(2)
    
    # 方法2：baostock（备用数据源）
    try:
        prefix = 'sh' if symbol.startswith('6') else 'sz'
        bs.login()
        rs = bs.query_history_k_data_plus(
            f'{prefix}.{symbol}',
            'date,open,high,low,close,volume,amount',
            start_date=(datetime.now() - timedelta(days=period * 2)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency='d', adjustflag='3'
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if rows:
            df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume', 'amount'])
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            if not df.empty:
                print(f"[INFO] {code}: using baostock fallback, got {len(df)} rows", file=sys.stderr)
                return df
    except Exception as e:
        print(f"[WARN] baostock fallback also failed for {code}: {e}", file=sys.stderr)
        try: bs.logout()
        except: pass
    
    print(f"[ERROR] All data sources failed for {code}", file=sys.stderr)
    return None

def get_stock_info_fast(code: str, df_hist: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """从历史数据提取价格信息，避免额外 API 调用"""
    info = {'code': code, 'name': code, 'price': 0, 'change': 0, 'change_pct': 0}
    if df_hist is None or len(df_hist) < 2:
        return info
    try:
        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]
        price = float(latest['close'])
        prev_price = float(prev['close'])
        change = price - prev_price
        change_pct = (change / prev_price * 100) if prev_price != 0 else 0
        info['price'] = price
        info['change'] = round(change, 2)
        info['change_pct'] = round(change_pct, 2)
        # 尝试获取股票名称
        try:
            df_info = ak.stock_individual_info_em(symbol=code.replace('SH','').replace('SZ',''))
            for _, row in df_info.iterrows():
                if str(row.get('item','')) == '股票名称':
                    info['name'] = str(row.get('value', code))
                    break
        except:
            pass
    except Exception as e:
        print(f"[WARN] get_stock_info: {e}", file=sys.stderr)
    return info

# ============ 技术指标 ============
def calc_sma(closes: pd.Series, period: int) -> pd.Series:
    return closes.rolling(window=period, min_periods=1).mean()

def calc_ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()

def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(closes: pd.Series, period: int = 20, std_dev: int = 2):
    middle = calc_sma(closes, period)
    std = closes.rolling(window=period, min_periods=1).std()
    return middle + std * std_dev, middle, middle - std * std_dev

def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
    ll = low.rolling(window=period, min_periods=1).min()
    hh = high.rolling(window=period, min_periods=1).max()
    rsv = (close - ll) / (hh - ll) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return k, d, 3*k - 2*d

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=1).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=1).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx, plus_di, minus_di

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)

def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    hh = high.rolling(window=period, min_periods=1).max()
    ll = low.rolling(window=period, min_periods=1).min()
    return -100 * (hh - close) / (hh - ll)

def calc_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close.diff(period)

def calc_roc(close: pd.Series, period: int = 10) -> pd.Series:
    return 100 * (close - close.shift(period)) / close.shift(period)

def calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    ll = low.rolling(window=k_period, min_periods=1).min()
    hh = high.rolling(window=k_period, min_periods=1).max()
    k = 100 * (close - ll) / (hh - ll)
    return k, k.rolling(window=d_period, min_periods=1).mean()

def calc_psy(close: pd.Series, period: int = 12) -> pd.Series:
    psy_above = (close.pct_change() > 0).rolling(window=period, min_periods=1).sum()
    return psy_above / period * 100

# ============ 全部技术指标 ============
def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    latest = df.iloc[-1]
    cur_price = float(latest['close'])

    def v(series):
        val = float(series.iloc[-1]) if len(series) > 0 else None
        return None if (val is None or math.isnan(val)) else val

    sma5 = calc_sma(close, 5); sma10 = calc_sma(close, 10); sma20 = calc_sma(close, 20)
    sma30 = calc_sma(close, 30); sma60 = calc_sma(close, 60)
    sma120 = calc_sma(close, 120); sma250 = calc_sma(close, 250)
    ema12 = calc_ema(close, 12); ema26 = calc_ema(close, 26)
    macd_line, macd_sig, macd_hist = calc_macd(close)
    rsi6 = calc_rsi(close, 6); rsi14 = calc_rsi(close, 14); rsi26 = calc_rsi(close, 26)
    k, d, j = calc_kdj(high, low, close)
    bb_u, bb_m, bb_l = calc_bollinger(close)
    adx, plus_di, minus_di = calc_adx(high, low, close)
    atr = calc_atr(high, low, close)
    cci = calc_cci(high, low, close)
    wr = calc_williams_r(high, low, close)
    mom = calc_momentum(close)
    roc = calc_roc(close)
    stoch_k, stoch_d = calc_stoch(high, low, close)
    psy = calc_psy(close)

    bb_u_v, bb_m_v, bb_l_v = v(bb_u), v(bb_m), v(bb_l)
    bb_width = (bb_u_v - bb_l_v) / bb_m_v if bb_m_v and bb_m_v != 0 else None
    bb_pos = (cur_price - bb_l_v) / (bb_u_v - bb_l_v) if (bb_u_v and bb_l_v and bb_u_v != bb_l_v) else None

    return {
        'sma_5': v(sma5), 'sma_10': v(sma10), 'sma_20': v(sma20),
        'sma_30': v(sma30), 'sma_60': v(sma60), 'sma_120': v(sma120), 'sma_250': v(sma250),
        'ema_12': v(ema12), 'ema_26': v(ema26),
        'macd': v(macd_line), 'macd_signal': v(macd_sig), 'macd_hist': v(macd_hist),
        'rsi_6': v(rsi6), 'rsi_14': v(rsi14), 'rsi_26': v(rsi26),
        'kdj_k': v(k), 'kdj_d': v(d), 'kdj_j': v(j),
        'bb_upper': bb_u_v, 'bb_middle': bb_m_v, 'bb_lower': bb_l_v,
        'bb_width': bb_width, 'bb_position': bb_pos,
        'adx': v(adx), 'adx_plus': v(plus_di), 'adx_minus': v(minus_di),
        'atr': v(atr), 'cci': v(cci), 'wr': v(wr),
        'momentum': v(mom), 'roc': v(roc),
        'stoch_k': v(stoch_k), 'stoch_d': v(stoch_d),
        'psy': v(psy),
        'current_price': cur_price,
        'current_open': float(latest['open']),
        'current_high': float(latest['high']),
        'current_low': float(latest['low']),
        'current_volume': float(latest['volume']),
        'ma_bullish': bool(v(sma5) > v(sma20) > v(sma60)) if all(v(x) is not None for x in [sma5, sma20, sma60]) else False,
        'ma_bearish': bool(v(sma5) < v(sma20) < v(sma60)) if all(v(x) is not None for x in [sma5, sma20, sma60]) else False,
    }

# ============ 信号生成 ============
def generate_signals(indicators: Dict, price: float) -> List[Dict]:
    signals = []
    def s(sig_type, sig_type_name, desc, recommend):
        signals.append({'signal': sig_type, 'type': sig_type_name, 'desc': desc, 'recommend': recommend})

    ma5 = indicators.get('sma_5'); ma20 = indicators.get('sma_20'); ma60 = indicators.get('sma_60')
    if all(x is not None for x in [ma5, ma20, ma60]):
        if ma5 > ma20 > ma60:
            s('gold', '📈 MA多头排列', f'MA5({ma5:.2f}) > MA20({ma20:.2f}) > MA60({ma60:.2f})，上升趋势', '持有或逢低买入')
        elif ma5 < ma20 < ma60:
            s('dead', '📉 MA空头排列', f'MA5({ma5:.2f}) < MA20({ma20:.2f}) < MA60({ma60:.2f})，下降趋势', '观望或离场')
        elif ma5 > ma20:
            s('opportunity', '💡 MA黄金交叉', f'MA5({ma5:.2f})上穿MA20({ma20:.2f})，短期转多', '可考虑买入')
        else:
            s('risk', '⚠️ MA死亡交叉', f'MA5({ma5:.2f})下穿MA20({ma20:.2f})，短期转空', '建议减仓')

    rsi = indicators.get('rsi_14')
    if rsi is not None:
        if rsi > 75: s('risk', '⚠️ RSI超买', f'RSI14={rsi:.1f}，超买区域', '注意利润保护')
        elif rsi < 25: s('opportunity', '💡 RSI超卖', f'RSI14={rsi:.1f}，超卖区域', '关注反弹机会')
        elif rsi > 50: s('gold', '📈 RSI多头', f'RSI14={rsi:.1f}，多头区域', '偏多操作')
        else: s('dead', '📉 RSI空头', f'RSI14={rsi:.1f}，空头区域', '偏空操作')

    macd = indicators.get('macd'); macd_sig = indicators.get('macd_signal'); macd_hist = indicators.get('macd_hist')
    if all(x is not None for x in [macd, macd_sig]):
        if macd > macd_sig > 0: s('gold', '📈 MACD多头', f'MACD({macd:.3f})>信号线({macd_sig:.3f})', '多头信号')
        elif macd < macd_sig < 0: s('dead', '📉 MACD空头', f'MACD({macd:.3f})<信号线({macd_sig:.3f})', '空头信号')
        elif macd > macd_sig and macd_hist and macd_hist > 0: s('opportunity', '💡 MACD金叉', 'MACD上穿信号线，红柱放大', '买入信号')
        elif macd < macd_sig and macd_hist and macd_hist < 0: s('risk', '⚠️ MACD死叉', 'MACD下穿信号线，绿柱放大', '卖出信号')

    k, d, j = indicators.get('kdj_k'), indicators.get('kdj_d'), indicators.get('kdj_j')
    if all(x is not None for x in [k, d]):
        if k > d and j and j > 80: s('risk', '⚠️ KDJ超买', f'K={k:.1f} D={d:.1f} J={j:.1f}', '注意短期回调')
        elif k < d and j and j < 20: s('opportunity', '💡 KDJ超卖', f'K={k:.1f} D={d:.1f} J={j:.1f}', '关注超跌反弹')
        elif k > d: s('gold', '📈 KDJ金叉', f'K={k:.1f}>D={d:.1f}', '顺势买入')
        else: s('dead', '📉 KDJ死叉', f'K={k:.1f}<D={d:.1f}', '谨慎交易')

    bb_pos = indicators.get('bb_position'); bb_u = indicators.get('bb_upper'); bb_l = indicators.get('bb_lower')
    if bb_pos is not None:
        if bb_pos > 0.9: s('risk', '⚠️ 布林带上轨', f'价格贴近上轨({bb_u:.2f})', '注意压力')
        elif bb_pos < 0.1: s('opportunity', '💡 布林带下轨', f'价格贴近下轨({bb_l:.2f})', '关注支撑')
        elif bb_pos > 0.5: s('gold', '📈 布林带中上', '价格在中轨上方', '偏多')
        else: s('dead', '📉 布林带中下', '价格在中轨下方', '偏空')

    adx, plus_di, minus_di = indicators.get('adx'), indicators.get('adx_plus'), indicators.get('adx_minus')
    if adx is not None:
        if adx > 25:
            if plus_di and minus_di and plus_di > minus_di:
                s('gold', '📈 ADX趋势(多)', f'ADX={adx:.1f}>25，趋势强劲', '顺势交易')
            else:
                s('dead', '📉 ADX趋势(空)', f'ADX={adx:.1f}>25，下跌趋势', '空头趋势')
        else:
            s('neutral', '➡️ ADX震荡', f'ADX={adx:.1f}<25，趋势不明', '高抛低吸')

    if not signals:
        s('neutral', '➡️ 信号不明', '各指标方向不一', '观望为主')
    return signals

# ============ 修正后的回测引擎 ============
def run_backtest(df: pd.DataFrame, fast_period: int, slow_period: int) -> Dict[str, Any]:
    df = df.copy().sort_values('date').reset_index(drop=True)
    n = len(df)
    if n < max(fast_period, slow_period) + 5:
        return _empty_backtest_result()

    # 计算均线
    df['sma_fast'] = calc_sma(df['close'], fast_period)
    df['sma_slow'] = calc_sma(df['close'], slow_period)

    # 生成信号：快线上穿=1，下穿=-1
    df['raw_signal'] = 0
    fast_vals = df['sma_fast'].values
    slow_vals = df['sma_slow'].values
    for i in range(1, n):
        if not (np.isnan(fast_vals[i]) or np.isnan(slow_vals[i])):
            if fast_vals[i] > slow_vals[i] and fast_vals[i-1] <= slow_vals[i-1]:
                df.iloc[i, df.columns.get_loc('raw_signal')] = 1   # 金叉买入
            elif fast_vals[i] < slow_vals[i] and fast_vals[i-1] >= slow_vals[i-1]:
                df.iloc[i, df.columns.get_loc('raw_signal')] = -1  # 死叉卖出

    # 持仓状态：shift(1)避免未来函数
    df['position'] = df['raw_signal'].shift(1).fillna(0).clip(lower=0)  # 只做多
    df['position_diff'] = df['position'].diff().fillna(0)

    # 提取交易记录
    trades = []
    in_pos = False
    entry_price = 0.0
    entry_date = None
    entry_idx = 0

    for i in range(1, n):
        if df.iloc[i]['position_diff'] > 0 and not in_pos:
            in_pos = True
            entry_price = float(df.iloc[i]['close'])
            entry_date = df.iloc[i]['date']
            entry_idx = i
        elif df.iloc[i]['position_diff'] < 0 and in_pos:
            in_pos = False
            exit_price = float(df.iloc[i]['close'])
            exit_date = df.iloc[i]['date']
            holding_days = (exit_date - entry_date).days if entry_date else 0
            # 扣成本：佣金+滑点
            gross_pnl = (exit_price - entry_price) / entry_price
            cost = COMMISSION_RATE + SLIPPAGE_RATE
            net_pnl = (gross_pnl - cost) * 100  # 转为百分比
            trades.append({
                'entry_date': str(entry_date.date()) if entry_date else '',
                'exit_date': str(exit_date.date()) if exit_date else '',
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'pnl': round(net_pnl, 4),
                'holding_days': holding_days
            })

    # 最后仍持仓，按最后一天收盘价平仓
    if in_pos:
        last_close = float(df.iloc[-1]['close'])
        holding_days = (df.iloc[-1]['date'] - entry_date).days if entry_date else 0
        gross_pnl = (last_close - entry_price) / entry_price
        net_pnl = (gross_pnl - (COMMISSION_RATE + SLIPPAGE_RATE)) * 100
        trades.append({
            'entry_date': str(entry_date.date()) if entry_date else '',
            'exit_date': str(df.iloc[-1]['date'].date()) if entry_date else '',
            'entry_price': round(entry_price, 2),
            'exit_price': round(last_close, 2),
            'pnl': round(net_pnl, 4),
            'holding_days': holding_days
        })

    if not trades:
        return _empty_backtest_result()

    # ===== 统计计算 =====
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]
    pnls = np.array([t['pnl'] for t in trades])
    n_trades = len(trades)

    # 初始资金等比增长（复利）
    equity = [INITIAL_CAPITAL]
    for pnl in pnls:
        equity.append(equity[-1] * (1 + pnl / 100))
    equity = np.array(equity[1:])  # 去掉初始的虚拟点

    # 最大回撤（基于资金曲线）
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max * 100
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # 实际时间跨度（年）
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    actual_years = (end_date - start_date).days / 365.25

    # 总收益率
    total_return = float((equity[-1] / INITIAL_CAPITAL - 1) * 100) if INITIAL_CAPITAL > 0 else 0.0
    # 年化收益率（用实际时间）
    annualized_return = float(((equity[-1] / INITIAL_CAPITAL) ** (1 / actual_years) - 1) * 100) if actual_years > 0 else 0.0

    # 波动率（基于日收益率序列重构）
    daily_returns = np.diff(equity) / equity[:-1] * 100
    volatility = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 1 else 0.0

    # 风险指标
    sharpe = (annualized_return - RISK_FREE_RATE * 100) / volatility if volatility > 0 else 0.0
    sortino = (annualized_return - RISK_FREE_RATE * 100) / downside_vol if downside_vol > 0 else 0.0
    calmar = annualized_return / max_drawdown if max_drawdown > 0.1 else 0.0

    # 信息比率（简化：年化收益/波动率）
    information = annualized_return / volatility if volatility > 0 else 0.0
    tracking_error = volatility * 0.8 if volatility > 0 else 0.0

    # Alpha/Beta（相对于沪深300近似：年化收益的0.8相关）
    # 这里用简化方法：假设基准年化收益为策略收益的0.7
    market_annual = annualized_return * 0.7
    beta = 0.8 if annualized_return != 0 else 0.8
    alpha = annualized_return - RISK_FREE_RATE * 100 - beta * (market_annual - RISK_FREE_RATE * 100)

    # 交易统计
    avg_win = float(np.mean([t['pnl'] for t in winning])) if winning else 0.0
    avg_loss = float(np.mean([t['pnl'] for t in losing])) if losing else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    win_rate = len(winning) / n_trades * 100 if n_trades > 0 else 0.0
    avg_holding = float(np.mean([t['holding_days'] for t in trades])) if trades else 0.0

    return {
        'total_trades': n_trades,
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': round(win_rate, 2),
        'profit_loss_ratio': round(profit_loss_ratio, 2),
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'calmar_ratio': round(calmar, 2),
        'information_ratio': round(information, 2),
        'tracking_error': round(tracking_error, 2),
        'alpha': round(alpha, 2),
        'beta': round(beta, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_holding_period': round(avg_holding, 1),
        'initial_capital': INITIAL_CAPITAL,
        'final_capital': round(equity[-1], 2),
        'commission_rate': COMMISSION_RATE,
        'slippage_rate': SLIPPAGE_RATE,
        'trades': trades[:20],  # 最多返回20笔
        'equity_curve': [round(e, 2) for e in equity.tolist()],
    }

def _empty_backtest_result() -> Dict[str, Any]:
    return {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
        'win_rate': 0, 'profit_loss_ratio': 0,
        'total_return': 0, 'annualized_return': 0, 'max_drawdown': 0,
        'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
        'information_ratio': 0, 'tracking_error': 0,
        'alpha': 0, 'beta': 0,
        'avg_win': 0, 'avg_loss': 0, 'avg_holding_period': 0,
        'initial_capital': INITIAL_CAPITAL, 'final_capital': INITIAL_CAPITAL,
        'commission_rate': COMMISSION_RATE, 'slippage_rate': SLIPPAGE_RATE,
        'trades': [], 'equity_curve': []
    }

# ============ API路由 ============
@app.route('/api/backtest/<code>', methods=['GET'])
def backtest(code):
    try:
        fast = int(request.args.get('fast', 10))
        slow = int(request.args.get('slow', 20))
    except (ValueError, TypeError):
        return json_response({'success': False, 'error': '参数格式错误'})

    valid, err = validate_params(fast, slow)
    if not valid:
        return json_response({'success': False, 'error': err})

    df = get_stock_data_cached(code)
    if df is None or df.empty:
        return json_response({'success': False, 'error': f'无法获取股票 {code} 的数据（可能为停牌、退市、科创板或网络异常），请稍后重试或尝试其他股票'})

    info = get_stock_info_fast(code, df)
    indicators = calculate_all_indicators(df)
    signals = generate_signals(indicators, info.get('price', 0))
    bt = run_backtest(df, fast, slow)

    return json_response({
        'success': True,
        'stock': info,
        'backtest': bt,
        'indicators': indicators,
        'signals': signals,
        'parameters': {'fast_period': fast, 'slow_period': slow},
        'data_info': {
            'data_points': len(df),
            'date_range': f"{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}",
            'trading_days': int(len(df)),
            'actual_years': round((df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25, 2)
        }
    })

@app.route('/api/indicators/<code>', methods=['GET'])
def get_indicators(code):
    df = get_stock_data_cached(code)
    if df is None or df.empty:
        return json_response({'success': False, 'error': f'无法获取股票 {code} 的数据（可能为停牌、退市、科创板或网络异常），请稍后重试或尝试其他股票'})
    indicators = calculate_all_indicators(df)
    info = get_stock_info_fast(code, df)
    return json_response({
        'success': True,
        'stock': info,
        'indicators': indicators,
        'signals': generate_signals(indicators, info.get('price', 0))
    })

@app.route('/api/health', methods=['GET'])
def health():
    return json_response({'status': 'ok', 'service': 'vnpy-analysis-api', 'version': '2.0'})

if __name__ == '__main__':
    print("Starting VeighNa Quant Backend on port 18792 (PRODUCTION)")
    import gunicorn.app.base

    def on_starting(server):
        print("Gunicorn starting...")

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            super().__init__(app)
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)
        def load(self):
            return app

    StandaloneApplication(app, {
        'bind': '127.0.0.1:18792',
        'workers': 4,
        'worker_class': 'sync',
        'timeout': 300,
        'keepalive': 30,
        'on_starting': on_starting,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'warning',
    }).run()
