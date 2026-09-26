import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\nguye\.gemini\antigravity-ide\brain\6c9dfd03-9fac-4352-817e-e5abf00b0549\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if 115 <= idx <= 150:
            data = json.loads(line)
            source = data.get('source')
            stype = data.get('type')
            if stype == 'USER_INPUT':
                print(f"[{idx}] USER: {data.get('content')[:100]}")
            elif source == 'MODEL' and stype == 'PLANNER_RESPONSE':
                for tc in data.get('tool_calls', []):
                    print(f"[{idx}] TOOL: {tc['name']} -> {str(tc['args'])[:120]}")
