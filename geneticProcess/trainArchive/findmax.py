import json

# Load JSON Lines
with open('archive_ans.json', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Find the dict with the max 'mem'
max_dict = max(data, key=lambda x: x.get('mem', float('-inf')))

# Print number of dicts and the one with max 'mem'
print(f"Total dictionaries: {len(data)}")
print("Dict with max 'mem':")
print(max_dict)
