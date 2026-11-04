"""Build a source-only publication bundle using an explicit allowlist."""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

FILES = ('README.md', 'pyproject.toml', 'CITATION.cff', '.gitignore',
         'environment.yml', 'requirements.txt')
DIRECTORIES = ('src', 'configs', 'scripts', 'tools', 'tests', 'docs',
               'examples', 'benchmarks', '.github')


def release_files(root):
    for name in FILES:
        yield root/name
    for name in DIRECTORIES:
        for path in sorted((root/name).rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink():
                continue
            if any(x == '__pycache__' or x.endswith('.egg-info') for x in relative.parts):
                continue
            if path.name.startswith('local_') or path.suffix in {'.pyc', '.ckpt', '.h5ad', '.npy', '.npz', '.pdf'}:
                continue
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError(args.output)
    files = list(release_files(root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=Path('scDVD')/path.relative_to(root), recursive=False)
    print(f'Exported {len(files)} files: {args.output}')


if __name__ == '__main__':
    main()
