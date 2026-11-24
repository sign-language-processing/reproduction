"""Compare project decoder indices/RGB bytes with the native OpenCV backend."""
from pathlib import Path
import json,sys,subprocess,shutil
import cv2,numpy as np
from simple_video_utils.frames import read_frames_exact
sys.path.insert(0,'/upstream/CiCo/I3D_feature_extractor')
from datasets.videodataset import VideoDataset
class ProbeDataset(VideoDataset):
 def _set_datasetname(self):pass
 def _get_nframes(self,ind):return 0
reader=ProbeDataset.__new__(ProbeDataset)
p=sorted(Path('/datasets/rwth-phoenix-2014-t/videos/train').glob('*.mp4'))[0]
local=Path('/tmp/decoder-audit.mp4');shutil.copyfile(p,local)
cap=cv2.VideoCapture(str(local));native=[]
while True:
 ok,frame=cap.read()
 if not ok:break
 native.append(frame[:,:,::-1])
cap.release();native=np.stack(native)
cases=[]
for start,end in [(0,15),(7,22),(len(native)-16,len(native)-1),(len(native)-5,len(native)+4)]:
 actual=np.stack(list(read_frames_exact(str(local),start,end)));expected=native[start:min(end+1,len(native))]
 assert actual.shape==expected.shape
 diff=np.abs(actual.astype('int16')-expected.astype('int16'))
 cases.append({'range_inclusive':[start,end],'frames':len(actual),'shape':list(actual.shape),'max_rgb_absolute_difference':int(diff.max()),'mean_rgb_absolute_difference':float(diff.mean()),'exact_bytes':bool(np.array_equal(actual,expected))})
short=Path('/tmp/decoder-short.mp4');subprocess.run(['ffmpeg','-v','error','-i',str(local),'-frames:v','8','-c:v','libx264','-y',str(short)],check=True)
short_frames=list(read_frames_exact(str(short),0,15));assert len(short_frames)==8
patched=reader._load_rgb(str(short),range(8)).reshape(3,16,260,210).numpy()
assert np.array_equal(patched[:,7],patched[:,15])
assert np.allclose(patched[:,0],short_frames[0].transpose(2,0,1)/255,atol=1e-7)
report={'native_import_and_patched_short_padding':True,'source':str(p),'native_frames':len(native),'cases':cases,'short_video_available_frames':len(short_frames),'endpoint_semantics':'inclusive','short_tail_policy':'repeat last valid decoded frame to16 as native extractor'}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2));print(json.dumps(report))
