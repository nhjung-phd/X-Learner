
import os, zipfile, time, hashlib
EXCLUDES = {".git","__pycache__","dist",".ipynb_checkpoints",".mypy_cache",".pytest_cache"}
def create_zip(root=".", out_dir="dist", name_prefix="xlearner_gui", excludes=None):
    excludes = set(excludes or []) | EXCLUDES
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name_prefix}_{ts}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, dnames, fnames in os.walk(root):
            if any(x in r for x in excludes): continue
            for f in fnames:
                p = os.path.join(r,f)
                if any(x in p for x in excludes): continue
                z.write(p, arcname=os.path.relpath(p, root))
    sha = hashlib.sha256(open(path,"rb").read()).hexdigest()
    with open(path + ".sha256","w") as s: s.write(sha)
    return os.path.abspath(path)
