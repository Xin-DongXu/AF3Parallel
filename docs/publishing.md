# Publishing AF3Parallel

Guide for releasing new versions to **PyPI**.

Current release: **v1.1.0** — https://pypi.org/project/af3parallel/1.1.0/

---

## 1. PyPI

### One-time setup

1. Account at [pypi.org](https://pypi.org/account/register/)
2. Authentication (choose one):

   **Trusted publishing (recommended)**

   PyPI → project `af3parallel` → Publishing → Add pending publisher:

   | Field | Value |
   | --- | --- |
   | Owner | `Xin-DongXu` |
   | Repository | `AF3Parallel` |
   | Workflow | `publish-pypi.yml` |
   | Environment | `pypi` |

   GitHub → Settings → Environments → create `pypi`.

   **API token**

   PyPI → API tokens → **Entire account** scope (required for first upload).

   GitHub secret: `PYPI_API_TOKEN` (if using twine in Actions).

### Release steps

```bash
pip install build twine
python -m build
twine check dist/*
```

```bash
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
# GitHub → Releases → Publish
```

Or manual upload:

```powershell
$env:TWINE_USERNAME = '__token__'
$env:TWINE_PASSWORD = (Get-Clipboard -Raw).Trim()
twine upload dist\*
```

Verify:

```bash
pip install af3parallel
af3parallel --version
```

---

## 2. Version bumps

Update together:

| File | Field |
| --- | --- |
| `src/af3parallel/__version__.py` | `__version__` |
| `pyproject.toml` | `version` |
| `CITATION.cff` | `version` |
| `CHANGELOG.md` | new section |

Then rebuild, upload, and tag.
