import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\nguye\.gemini\antigravity-ide\brain\6c9dfd03-9fac-4352-817e-e5abf00b0549\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx == 491:
            data = json.loads(line)
            print(f"Step 491 CONTENT:\n{data.get('content')}")
