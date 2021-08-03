# https://github.com/e-mon/lish-moa/blob/master/encode.py
import base64
import git
import gzip
from pathlib import Path

template = """
import gzip
import base64
import os
from pathlib import Path
from typing import Dict
# this is base64 encoded source code
file_data: Dict = {file_data}
for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))
# output current commit hash
print('{commit_hash}')
"""

def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def build_script():
    to_encode = list(Path('src').glob('**/*.py'))
    file_data = {str(path).replace('\\', '/'): encode_file(path) for path in to_encode}
    output_path = Path('.build/script.py')
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(template.replace('{file_data}', str(file_data)).replace('{commit_hash}', get_current_commit_hash()), encoding='utf8')


if __name__ == '__main__':
    build_script()
