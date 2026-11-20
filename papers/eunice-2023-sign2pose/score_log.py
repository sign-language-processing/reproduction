"""Test accuracies from a SPOTER log: SPOTER's max-over-test pick, the validation-selected checkpoint, macro accuracy."""
import ast, re, sys
text = open(sys.argv[1]).read()
# Restarts append to the same log. Without resume (patches/0004) a restart re-runs from epoch 1, so use the
# final attempt; with resume, later segments continue earlier ones, so key by epoch (last occurrence wins).
if "Resumed after epoch" not in text:
    text = text[text.rfind("Starting "):]
val = list(dict((int(e), float(v)) for e, v in re.findall(r"\[(\d+)\] VALIDATION  acc: ([0-9.eE-]+)", text)).values())
text = text[text.rfind("Testing checkpointed"):]
assert len(val) == 300, len(val)
# Replay train.py's windows: save on strict improvement, reset top_val after epochs with epoch % 10 == 0.
best, idx, top = {}, 0, 0.0
for e, v in enumerate(val):
    if v > top:
        top, best[idx] = v, (v, e + 1)
    if e % 10 == 0:
        top, idx = 0.0, idx + 1
ckpt = max(best, key=lambda k: (best[k][0], -k))  # highest validation accuracy, earliest window on ties
test, stats = {}, None
for line in text.splitlines():
    if "[INFO] {" in line:
        stats = ast.literal_eval(line.split("[INFO] ", 1)[1])
    m = re.search(r"(checkpoint_[tv]_\d+)  ->  ([0-9.]+)", line)
    if m and "[INFO]" in line:
        test[m.group(1)] = (float(m.group(2)), sum(stats.values()) / len(stats), len(stats))
top_name = re.search(r"The best checkpoint is .*/(checkpoint_[tv]_\d+)", text).group(1)
v = f"checkpoint_v_{ckpt}"
print(f"{sys.argv[1]}: {len(test)} checkpoints tested")
print(f"  SPOTER max-over-test : {top_name} per-instance {100*test[top_name][0]:.2f}  macro {100*test[top_name][1]:.2f}")
print(f"  validation-selected  : {v} (val {100*best[ckpt][0]:.2f} @ epoch {best[ckpt][1]}) per-instance {100*test[v][0]:.2f}  macro {100*test[v][1]:.2f}  classes in test {test[v][2]}")
