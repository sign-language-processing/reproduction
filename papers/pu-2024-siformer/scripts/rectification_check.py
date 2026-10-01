"""Exercise the recovered author FE-then-AA recipe on four real LSA64 records."""
import ast, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/opt/siformer/rectifications')
from flexion_extension_rectification import rectify_finger_flexion_and_extension
from abduction_adduction_rectification import rectify_finger_abduction_and_addiction

out=Path(sys.argv[1])
indices=[0,50,100,150]
with Path('/datasets/lsa64/siformer/LSA64_60fps.csv').open() as f:
    reader=csv.DictReader(f);rows=[]
    for index,row in enumerate(reader):
        if index in indices:rows.append(row)
        if index==indices[-1]:break
pd.DataFrame(rows).to_csv(out/'input.csv',index=False)
revision='09a7c1c575849edbd5e245dc6d1c80ddce94188a'
for name in ['abduction_adduction_ranges.csv','flexion_extension_ranges.csv']:
    (out/name).write_bytes(subprocess.check_output(['git','-C','/opt/siformer','show',revision+':active_motion/'+name]))
fe=rectify_finger_flexion_and_extension(str(out/'input.csv'),str(out/'flexion_extension_ranges.csv'),alpha=0.4)
fe.to_csv(out/'fe.csv',index=False)
aafe=rectify_finger_abduction_and_addiction(str(out/'fe.csv'),str(out/'abduction_adduction_ranges.csv'),alpha=0.4)
aafe.to_csv(out/'aafe.csv',index=False)
keys=[k for k,v in rows[0].items() if v.startswith('[')]
before=np.array([[ast.literal_eval(row[k]) for k in keys] for row in rows],dtype=float)
after=np.array([[aafe.at[i,k] for k in keys] for i in range(len(rows))],dtype=float)
assert before.shape==after.shape
assert np.isfinite(after).all()
result={'record_indices':indices,'labels':[r['labels'] for r in rows],'shape':list(after.shape),
        'all_finite':True,'changed_coordinate_values':int(np.count_nonzero(before!=after)),
        'alpha':0.4,'order':['flexion_extension','abduction_adduction'],
        'source_commit':'979a14ed15ed0f20afd77d447ad23c4f4107a2c3','motion_table_commit':revision,
        'historical_order_commit':'a6b3cb84c508057fbca71b19b9fcdd9d0443f6a7',
        'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix=='.csv'}}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
