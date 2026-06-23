import urllib.request, json

BASE = 'http://127.0.0.1:5001'

def t(ep, method='GET', data=None):
    url = BASE + ep
    try:
        body_data = json.dumps(data).encode() if data else None
        headers = {'Content-Type': 'application/json'} if data else {}
        req = urllib.request.Request(url, data=body_data, headers=headers)
        r = urllib.request.urlopen(req, timeout=60)
        body = r.read().decode('utf-8')
        try:
            parsed = json.loads(body)
            label = f'JSON keys={list(parsed.keys())[:8]}'
            if 'success' in parsed:
                label += f' success={parsed["success"]}'
                if parsed.get('success'):
                    m = parsed.get('metrics', {})
                    label += f' | ret={m.get("total_return")}% trades={m.get("total_trades")}'
            elif isinstance(parsed, list):
                label += f' len={len(parsed)}'
        except:
            label = f'{len(body)} bytes'
        print(f'  OK  [{r.status}] {ep} -> {label}')
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')
        print(f'  ERR [{e.code}] {ep} -> {err_body[:200]}')
    except Exception as e:
        print(f'  FAIL {ep} -> {str(e)[:120]}')

print('=== ETF 回测 (510300) ===')
t('/api/backtest', 'POST', {
    'symbol': '510300', 'asset_type': 'etf',
    'strategy': 'ma_crossover', 'params': {'short': 5, 'long': 20},
    'start': '2025-01-01', 'end': '2025-06-20', 'capital': 100000
})

print('\n=== 基金回测 (110022) ===')
t('/api/backtest', 'POST', {
    'symbol': '110022', 'asset_type': 'fund',
    'strategy': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9},
    'start': '2024-06-01', 'end': '2025-06-20', 'capital': 100000
})

print('\n=== ETF 策略对比 ===')
t('/api/compare', 'POST', {
    'symbol': '510300', 'asset_type': 'etf',
    'start': '2025-01-01', 'end': '2025-06-20', 'capital': 100000
})

print('\n=== 基金策略对比 ===')
t('/api/compare', 'POST', {
    'symbol': '110022', 'asset_type': 'fund',
    'start': '2024-06-01', 'end': '2025-06-20', 'capital': 100000
})
