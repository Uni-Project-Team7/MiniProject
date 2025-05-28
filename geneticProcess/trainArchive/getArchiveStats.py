import json
import os
absolute_path = os.path.abspath('./archive.json')
import sys
sys.path.append(os.path.abspath('../../'))
from geneticProcess.getMetrics.getAllStats import get_stats
import json
import torch
with open(absolute_path, 'r') as f:
    archive = [json.loads(line) for line in f]


def worker(candidate, ind, save_dict) :
    get_stats(candidate)
    save_dict.append(candidate)

save_dict = []
for ind, candidate in enumerate(archive):
    worker(candidate, ind, save_dict)
    with open('archive_ans.json', 'a') as f:
        f.write(json.dumps(candidate, separators=(', ', ': ')) + '\n')

