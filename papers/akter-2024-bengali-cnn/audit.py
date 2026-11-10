"""Read-only audit of the paper's public dataset sources; never trains on samples."""
import collections
import hashlib
import io
import json
import urllib.request
import zipfile


def read(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


card = json.loads(read('https://www.kaggle.com/api/v1/datasets/view/ksanzid/esharalipi-bangla-sign-language-dataset'))
print('Kaggle card:', json.dumps({k: card.get(k) for k in ['title', 'totalBytes', 'currentVersionNumber', 'licenseName', 'description']}))
url = 'https://www.kaggle.com/api/v1/datasets/download/ksanzid/esharalipi-bangla-sign-language-dataset?datasetVersionNumber=2'
archive = read(url)
files = zipfile.ZipFile(io.BytesIO(archive)).namelist()
print('Kaggle v2 archive SHA256:', hashlib.sha256(archive).hexdigest())
print('Kaggle v2 image paths:', json.dumps([p for p in files if p.lower().endswith(('.jpg', '.jpeg', '.png'))]))
for url in ['https://isharalipi.sanzidscloud.com', 'https://www.kaggle.com/api/v1/datasets/view/mmsabid/ishara-lipi-bangla-sign-language-dataset']:
    try:
        response = read(url)
        print('Source accessible:', url, len(response))
    except Exception as error:
        print('Source unavailable:', url, type(error).__name__, str(error))
mirror = 'https://raw.githubusercontent.com/cloudy4next/Ishara-Lipi/6888dce6c19332a8d282c2b47b59fbf073b25a39/dataset/database-20200430T230808Z-001.zip'
archive = read(mirror)
files = zipfile.ZipFile(io.BytesIO(archive)).namelist()
images = [p for p in files if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
print('Unverified 2020 mirror SHA256:', hashlib.sha256(archive).hexdigest())
print('Unverified mirror total:', len(images))
print('Unverified mirror class counts:', json.dumps(dict(sorted(collections.Counter(p.split('/')[1] for p in images).items()))))
print('Decision: exact paper-selected source images and augmented 7200/2160 split are not identified by these public artifacts. Do not train on a substitute.')
