#!/usr/bin/env bash
# Upload built distributions to PyPI.
#
# Prerequisites:
#   pip install build twine
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...   # API token from pypi.org
#
# Usage:
#   ./tools/publish-pypi.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python -m pip install --upgrade build twine
python -m build
twine check dist/*

if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "error: set TWINE_USERNAME=__token__ and TWINE_PASSWORD before uploading" >&2
  echo "       Or publish via GitHub Release (see docs/publishing.md)" >&2
  exit 1
fi

twine upload dist/*

echo ""
echo "Published. Verify with: pip install af3parallel && af3parallel --version"
python -c "
import hashlib, pathlib, glob
for p in sorted(glob.glob('dist/*.tar.gz')):
    h = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    print(f'sdist sha256 ({p}): {h}')
"
