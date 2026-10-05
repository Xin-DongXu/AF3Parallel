# Publishing AF3Parallel

Guide for releasing new versions to **PyPI**.

Current release: **v1.1.0** — https://pypi.org/project/af3parallel/1.1.0/

The GitHub Release tag **must** match the package version in
`pyproject.toml` and `src/af3parallel/__version__.py` (for example tag
`v1.1.0` for package `1.1.0`). A GitHub Release named `v1.0.0` will not
publish package `1.1.0`.

---

## 1. PyPI

### One-time setup

1. Account at [pypi.org](https://pypi.org/account/register/)
2. Authentication (choose one):

   **Trusted publishing (required by `.github/workflows/publish-pypi.yml`)**

   PyPI → project `af3parallel` → Publishing → Add a new pending publisher:

   | Field | Value |
   | --- | --- |
   | Owner | `Xin-DongXu` |
   | Repository | `AF3Parallel` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |

   Every field must match exactly. The failed run
   [AF3Parallel v1.0.0](https://github.com/Xin-DongXu/AF3Parallel/actions/runs/34177964331)
   was `invalid-publisher`: GitHub issued a valid OIDC token, but PyPI had
   no publisher with these claims.

   GitHub → Settings → Environments → create `pypi` (no protection rules
   required).

   **API token (manual / fallback)**

   PyPI → API tokens → project-scoped token for `af3parallel`.

   Then upload locally instead of using Actions:

   ```powershell
   $env:TWINE_USERNAME = '__token__'
   $env:TWINE_PASSWORD = (Get-Clipboard -Raw).Trim()
   twine upload dist\*
   ```

### Release steps

1. Bump versions together (`pyproject.toml`, `__version__.py`,
   `CITATION.cff`, `CHANGELOG.md`).
2. Merge to `main`.
3. Tag the commit that contains that version **and** the current
   publish workflow:

   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0"
   git push origin v1.1.0
   ```

4. GitHub → Releases → create a release from that tag → Publish.
   Publishing the release triggers `publish-pypi.yml`.

The workflow skips files that already exist on PyPI (`skip-existing`),
so re-releasing the current PyPI version after Trusted Publishing is
configured will succeed instead of failing with "file already exists".

Verify:

```bash
pip install -U af3parallel
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

---

## 3. Troubleshooting

### `invalid-publisher` / Trusted publishing exchange failure

PyPI did not find a Trusted Publisher that matches the GitHub OIDC
claims. For this repository the claims are:

- repository: `Xin-DongXu/AF3Parallel`
- workflow: `publish-pypi.yml`
- environment: `pypi`

Fix: add the publisher table above, wait a minute, then create a **new**
GitHub Release from a tag on `main` that includes this workflow. Do not
expect a re-run of an old tag (such as `v1.0.0`) to pick up workflow
edits made later on `main`.

### File already exists

`1.1.0` is already on PyPI. The workflow now uses `skip-existing: true`.
To publish new code, bump the version first. Do not reuse a lower tag
such as `v1.0.0` for a `1.1.0` package.
