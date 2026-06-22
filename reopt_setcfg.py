#!/usr/bin/env python3
"""Patch ladder knobs in the deployed /tmp/reopt_repo config.pbtxt files.

Usage: reopt_setcfg.py key=val key=val ...
Each key is routed to its owning model's config. Updates the value in-place
(the parameters block already contains the key from the S0 setup).
"""
import sys, re

OWNER = {
    "token_hop_len": "cosyvoice3", "flow_pre_lookahead_len": "cosyvoice3",
    "dynamic_chunk_strategy": "cosyvoice3", "enable_trim": "cosyvoice3",
    "prompt_feat_fp16": "cosyvoice3", "llm_seed": "cosyvoice3",
    "load_spk2info": "cosyvoice3",
    "flow_precision": "token2wav", "flow_trt": "token2wav",
    "hift_plan": "vocoder",
}
BASE = "/tmp/reopt_repo"


def set_param(model, key, val):
    p = f"{BASE}/{model}/config.pbtxt"
    s = open(p).read()
    # match:  key: "<key>", ... value: {string_value:"<old>"}
    pat = re.compile(r'(key:\s*"%s"\s*,\s*value:\s*\{string_value:")[^"]*("\})' % re.escape(key))
    if not pat.search(s):
        raise SystemExit(f"key {key} not found in {p}")
    s2 = pat.sub(lambda m: m.group(1) + val + m.group(2), s)
    open(p, "w").write(s2)


def main():
    changes = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        if k not in OWNER:
            raise SystemExit(f"unknown key {k}")
        set_param(OWNER[k], k, v)
        changes.setdefault(OWNER[k], []).append(f"{k}={v}")
    for m, c in changes.items():
        print(f"  {m}: {', '.join(c)}")


if __name__ == "__main__":
    main()
