"""Exercise the running HTTP service with three frozen, representative test rows."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    service = Path(__file__).resolve().parents[1]
    root = service.parent

    def call(path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf8')
        request = Request(args.url+path, data=data,
                          headers={'Content-Type': 'application/json; charset=utf-8'})
        try:
            with urlopen(request, timeout=180) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    for attempt in range(120):
        try:
            status, info = call('/health/ready')
            if status == 200:
                break
        except (URLError, OSError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError('Model service did not become ready')
    rows = [json.loads(line) for line in
            (root/'data/processed/two_events_v1/test.jsonl').read_text(encoding='utf8').splitlines()
            if line.strip()]
    selected = [next(row for row in rows if row['stratum'] == label)
                for label in ['企业收购', '股份回购', 'negative']]
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(call, '/v1/extract', {'text': selected[0]['text']})
        observed_busy = False
        for attempt in range(100):
            if first.done():
                break
            if call('/v1/model')[1]['busy']:
                observed_busy = True
                break
            time.sleep(0.05)
        assert observed_busy, 'Did not observe running inference'
        assert call('/health/live')[0] == 200
        busy_status, busy_body = call('/v1/extract', {'text': '并发请求'})
        assert busy_status == 503 and busy_body['error']['code'] == 'model_busy'
        status, body = first.result()
        assert status == 200 and body['valid_json_schema'], body
        results.append({'id': selected[0]['id'], 'reference': selected[0]['target'], 'response': body})
    for row in selected[1:]:
        status, body = call('/v1/extract', {'text': row['text']})
        assert status == 200 and body['valid_json_schema'], body
        results.append({'id': row['id'], 'reference': row['target'], 'response': body})
    status, body = call('/v1/extract', {'text': '中文'*3000})
    assert status == 422 and body['error']['code'] == 'input_too_long'
    assert call('/v1/extract', {'text': '   '})[0] == 422
    output = service/'run/smoke_result.json'
    output.parent.mkdir(exist_ok=True)
    result = {'passed': True, 'device': info['device'], 'dtype': info['dtype'],
              'examples': results, 'busy_returns_503': True, 'health_responsive_during_inference': True,
              'oversized_input_returns_422': True,
              'note': 'Three-row CPU HTTP smoke check, not a full125 benchmark rerun.'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'passed': True, 'examples': len(results),
                      'latency_ms': [r['response']['usage']['latency_ms'] for r in results]}))


if __name__ == '__main__':
    main()
