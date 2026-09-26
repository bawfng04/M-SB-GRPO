import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\nguye\.gemini\antigravity-ide\brain\6c9dfd03-9fac-4352-817e-e5abf00b0549\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            print(f"Step {idx} USER: {data.get('content')[:140]}")
