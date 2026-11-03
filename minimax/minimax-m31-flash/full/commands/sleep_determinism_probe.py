"""Persist independent sleep lifecycle steps and deterministic request evidence."""
import argparse
import base64
import datetime
import json
import pathlib
import sys
import traceback
import urllib.request
import urllib.error
from types import SimpleNamespace

here = pathlib.Path(__file__).resolve().parent
scripts = here.parents[1] / 'workflow/rl'
sys.path[:0] = [str(scripts), str(scripts/'weight-update'), str(scripts/'weight-update/common')]
from common.dual_api import ENDPOINTS, generate_endpoint, validate_generation_output
from rl_request_adapter import adapt_request
from test_reload_weights import reload_checkpoint_weights

parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=['capture','sleep','wake-weights','reload','wake-kv'], required=True)
parser.add_argument('--level', type=int, choices=[1,2], required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--cold-prefix', action='store_true')
args = parser.parse_args()
path = pathlib.Path(args.report)
report = json.loads(path.read_text()) if path.exists() else {
    'level': args.level, 'status': 'running', 'steps': [], 'captures': {},
    'sampling': {'temperature':0,'top_p':1,'top_k':-1,'min_p':0,'seed':0,
                 'frequency_penalty':0,'presence_penalty':0,'repetition_penalty':1,'max_tokens':2048},
    'requests_serial': True,
}
assert report['level'] == args.level
base = 'http://127.0.0.1:31070'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
evidence = []
active_dp_rank = None

def request(base, method, endpoint, payload=None, *, timeout=3600, headers=None):
    if payload is not None and endpoint.startswith('/v1/'):
        payload = dict(payload)
        payload.update(report['sampling'])
        if args.cold_prefix:
            state = request(base,'POST','/reset_prefix_cache')
            assert state.get('success') is True, state
    body = None if payload is None else json.dumps(payload).encode()
    request_headers={'Content-Type':'application/json',**(headers or {})}
    if endpoint in ENDPOINTS and active_dp_rank is not None:
        request_headers['X-data-parallel-rank']=str(active_dp_rank)
    req = urllib.request.Request(base+endpoint,data=body,method=method,headers=request_headers)
    try:
        response=opener.open(req,timeout=timeout)
    except urllib.error.HTTPError as error:
        evidence.append({'method':method,'endpoint':endpoint,'headers':request_headers,'payload':payload,
            'http_status':error.code,'error_body':error.read().decode(errors='replace')})
        raise
    with response:
        raw = response.read()
        data = json.loads(raw) if raw.strip() else None
        evidence.append({'method':method,'endpoint':endpoint,'headers':request_headers,'payload':payload,'http_status':response.status,'response':data})
        return data

def capture_replica(model, dp_rank, result):
    global active_dp_rank
    active_dp_rank = dp_rank
    for endpoint in ENDPOINTS:
        for index,prompt in enumerate(['Reply with exactly the word Hello.','What is 2 + 3? Answer briefly.']):
            result[f'dp{dp_rank}:'+endpoint+':'+str(index)] = generate_endpoint(adapt_request(request),base,model,prompt,2048,endpoint=endpoint,timeout=3600)
    image = (here/'assets/red-square-blue-circle.png').read_bytes()
    data = request(base,'POST','/v1/chat/completions',{
        'model':model,'messages':[{'role':'user','content':[
            {'type':'text','text':'Describe the shapes and their colors in this image.'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}}
        ]}],'logprobs':True,'top_logprobs':1,
    })
    choice = data['choices'][0]
    details = choice['logprobs']['content']
    output = {'text':choice['message']['content'],'tokens':[x['token'] for x in details],
              'token_logprobs':[x['logprob'] for x in details]}
    validate_generation_output('image_chat',output)
    result[f'dp{dp_rank}:image_chat'] = output
    return result

def capture():
    result={}
    report['captures'][args.label] = {'outputs':result,'cold_prefix_reset_each_request':args.cold_prefix,'requests':evidence}
    model = request(base,'GET','/v1/models')['data'][0]['id']
    world = request(base,'GET','/get_world_size?include_dp=true')['world_size']
    group = request(base,'GET','/get_world_size?include_dp=false')['world_size']
    assert world==16 and group==4, (world,group)
    report['cluster']={'world_size':world,'workers_per_dp':group,'data_parallel_size':4,
                       'generation_routing':'X-data-parallel-rank','expected_nodes':2}
    for dp_rank in range(4):capture_replica(model,dp_rank,result)
    for row in evidence:
        if row['endpoint'] in ENDPOINTS:
            assert row['response']['choices'][0]['finish_reason'] != 'length', 'Truncated validation request'
    report['captures'][args.label] = {'outputs':result,'cold_prefix_reset_each_request':args.cold_prefix,'requests':evidence}

step = {'label':args.label,'action':args.action,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    if args.action == 'capture': capture()
    elif args.action == 'sleep':
        request(base,'POST',f'/sleep?level={args.level}&mode=abort')
        assert request(base,'GET','/is_sleeping')['is_sleeping'] is True
    elif args.action == 'wake-weights':
        request(base,'POST','/wake_up?tags=weights')
        assert request(base,'GET','/is_sleeping')['is_sleeping'] is True
    elif args.action == 'reload':
        assert args.level == 2, 'Level1 must restore CPU backup without checkpoint reload'
        reload_checkpoint_weights(SimpleNamespace(server_url=base,updated_model='/path/to/artifact',timeout=3600))
    elif args.action == 'wake-kv':
        request(base,'POST','/wake_up?tags=kv_cache')
        assert request(base,'GET','/is_sleeping')['is_sleeping'] is False
        request(base,'GET','/health')
    step['status'] = 'passed'
except Exception:
    step.update(status='failed',traceback=traceback.format_exc())
    report['status'] = 'failed'
finally:
    step['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.action != 'capture': step['requests'] = evidence
    report['steps'].append(step)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(step))
raise SystemExit(0 if step['status']=='passed' else 1)
