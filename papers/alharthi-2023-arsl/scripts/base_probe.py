"""Test the study base image before selecting the paper's older TF stack."""
import modal
app = modal.App('arsl-4982792c-base-probe')
image = modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
@app.function(image=image, timeout=300)
def probe():
    import subprocess,sys
    print('PYTHON',sys.version)
    p=subprocess.run([sys.executable,'-m','pip','install','--dry-run','--no-deps','tensorflow==2.8.0'],capture_output=True,text=True)
    print(p.stdout,p.stderr)
    return {'python':sys.version,'exit_code':p.returncode,'image_id':image.object_id}
@app.local_entrypoint()
def main():
    print(probe.remote())
