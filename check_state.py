import json
import os

state_path = 'compare/A-MSB-GRPO-E/checkpoints/epoch1_step13763/training_state.json'
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
    print(f"Epoch: {state.get('epoch')}")
    print(f"Global Step: {state.get('global_step')}")
else:
    print("No training state found")
