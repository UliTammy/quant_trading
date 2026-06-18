import urllib.request, json

BASE = 'http://127.0.0.1:5001'

def test(ep, method='GET', data=None):
    url = BASE + ep
    try:
        body_data = json.dumps(data).encode() if data else None
        headers = {'Content-Type': 'application/json'} if data else {}
        req = urllib.request.Request(url, data=body_data, headers=headers)
        r = urllib.request.urlopen(req, timeout=30)
        body = r.read()
        try:
            parsed = json.loads(body)
            label = f'JSON, keys={list(parsed.keys())[:8]}'
            if 'success' in parsed:
                label += f' success={parsed["success"]}'
        except:
            label = f'{len(body)} bytes HTML'
        print(f'  OK  [{r.status}] {ep} -> {label}')
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')
        print(f'  ERR [{e.code}] {ep} -> {err_body[:150]}')
    except Exception as e:
        print(f'  FAIL {ep} -> {str(e)[:120]}')

print('=== 1. Page (/) ===')
test('/')

print('\n=== 2. Strategies API ===')
test('/api/strategies')

print('\n=== 3. Search API ===')
test('/api/search_stock?q=600')

print('\n=== 4. Backtest API ===')
test('/api/backtest', 'POST', {
    'symbol': '600519', 'strategy': 'ma_crossover',
    'params': {'short': 5, 'long': 20},
    'start': '2024-01-01', 'end': '2025-06-13', 'capital': 100000
})

print('\n=== 5. Compare API ===')
test('/api/compare', 'POST', {
    'symbol': '600519', 'start': '2024-01-01',
    'end': '2025-06-13', 'capital': 100000
})
