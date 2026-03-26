"""Patch torch.load calls to use weights_only=False (PyTorch >=2.6 changed default)."""
import os
import re

def patch_file(path):
    with open(path) as f:
        content = f.read()
    if "torch.load(" not in content:
        return
    new = re.sub(
        r"torch\.load\(([^)]*map_location[^)]*)\)",
        lambda m: m.group(0) if "weights_only" in m.group(0)
                  else "torch.load(" + m.group(1) + ", weights_only=False)",
        content,
    )
    if new != content:
        with open(path, "w") as f:
            f.write(new)
        print(f"Patched: {path}")

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fname in files:
        if fname.endswith(".py"):
            patch_file(os.path.join(root, fname))
