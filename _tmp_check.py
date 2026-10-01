import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os, random, warnings

nb = json.loads(Path(r'c:/Users/KIIT/OneDrive/Desktop/Brain Tumour/brain_mri_classification.ipynb').read_text(encoding='utf-8'))
ns = {}
ns['display'] = lambda *args, **kwargs: None
ns['Path'] = Path
ns['np'] = np
ns['pd'] = pd
ns['cv2'] = cv2
ns['plt'] = plt
ns['sns'] = sns
ns['Image'] = Image
ns['warnings'] = warnings
ns['os'] = os
ns['random'] = random

for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.strip():
        continue
    print(f'Running cell {idx}: {src.splitlines()[0][:80]}')
    try:
        exec(src, ns, ns)
        print('OK')
    except Exception as exc:
        print('ERROR in cell', idx, type(exc).__name__, exc)
        raise
print('ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY')
