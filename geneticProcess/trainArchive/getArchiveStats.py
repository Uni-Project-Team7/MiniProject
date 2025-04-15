import json
import os
absolute_path = os.path.abspath('./archive.json')
import sys
sys.path.append(os.path.abspath('../../'))
from geneticProcess.getMetrics.getAllStats import get_stats
import json

with open(absolute_path, 'r') as f:
    archive = [json.loads(line) for line in f]


def worker(candidate, device, ind) :
    get_stats(candidate, device)
    store_path = os.path.abspath('temp_candidates')
    with open(store_path + str(ind) + ".json", "w") as f:
        json.dump(candidate, f)

for ind, candidate in enumerate(archive):
    worker(candidate, 'cuda:0', ind)