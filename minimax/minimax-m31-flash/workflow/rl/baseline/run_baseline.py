#!/usr/bin/env python3
"""Generate and persist a reusable baseline completion result."""
import argparse, json, shlex, sys, urllib.request
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
from common.feature_arguments import (
    add_generation_arguments,
    add_report_arguments,
    add_service_arguments,
    create_feature_parser,
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = create_feature_parser(__doc__)
    add_service_arguments(parser, model_option="--model", require_model=True)
    add_generation_arguments(parser, prompt_input="repeat")
    add_report_arguments(parser, report_option="--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    a=parse_args(argv); a.prompts = a.prompts or ["Please explain in detail how distributed inference systems coordinate data parallelism, tensor parallelism, expert parallelism, weight updates, and service recovery while preserving deterministic outputs across repeated validation runs."]; outputs=[]; chat_outputs=[]
    for prompt in a.prompts:
        body=json.dumps({"model":a.model,"prompt":prompt,"max_tokens":a.max_tokens,"temperature":0,"top_p":1,"seed":0}).encode()
        req=urllib.request.Request(a.server_url.rstrip("/")+"/v1/completions", data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=600) as r: data=json.load(r)
        choice=(data.get("choices") or [{}])[0]; outputs.append({"prompt":prompt,"text":choice.get("text", ""),"finish_reason":choice.get("finish_reason")})
        chat_body=json.dumps({"model":a.model,"messages":[{"role":"user","content":prompt}],"max_tokens":a.max_tokens,"temperature":0,"top_p":1,"seed":0}).encode()
        chat_req=urllib.request.Request(a.server_url.rstrip("/")+"/v1/chat/completions", data=chat_body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(chat_req, timeout=600) as r: chat_data=json.load(r)
        chat_choice=(chat_data.get("choices") or [{}])[0]; message=chat_choice.get("message") or {}
        chat_outputs.append({"prompt":prompt,"text":message.get("content", ""),"finish_reason":chat_choice.get("finish_reason")})
    result={"mode":"baseline","server_url":a.server_url,"model":a.model,"prompts":a.prompts,"max_tokens":a.max_tokens,"outputs":outputs,"outputs_by_endpoint":{"/v1/completions":outputs,"/v1/chat/completions":chat_outputs},"endpoint_checks":{"/v1/completions":True,"/v1/chat/completions":True}}
    if a.verification_command:
        result["verification_command"] = a.verification_command
    else:
        result["verification_command"] = shlex.join([sys.executable, *sys.argv])
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(path)
    commands = path.parent / "launch_commands.txt"
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"VERIFIER: {result['verification_command']}\n")
if __name__=="__main__": main()
