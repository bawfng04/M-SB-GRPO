import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\nguye\.gemini\antigravity-ide\brain\6c9dfd03-9fac-4352-817e-e5abf00b0549\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx >= 450:
            data = json.loads(line)
            stype = data.get('type')
            source = data.get('source')
            if stype == 'USER_INPUT':
                print(f"[{idx}] USER: {data.get('content')[:100]}")
            elif source == 'MODEL' and stype == 'PLANNER_RESPONSE':
                content = data.get('content', '')
                tcs = data.get('tool_calls', [])
                if content:
                    print(f"[{idx}] MODEL CONTENT: {content[:150]}...")
                if tcs:
                    print(f"[{idx}] TOOL CALLS: {[tc['name'] for tc in tcs]}")
