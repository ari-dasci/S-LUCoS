# Repository Setup Scripts

This folder contains setup helpers for local dependencies that are required by
the LUCoS experiments but are maintained as Git submodules rather than ordinary
PyPI packages.


## Files

```text
scripts/
|--- setup_external_libs.sh
`--- external_libs/
    |--- RDSS.pyproject.toml
    |--- TabClustPFN.pyproject.toml
    `--- zcore.pyproject.toml
```

## `setup_external_libs.sh`

Run this script after creating and activating the Python environment:

```bash
chmod +x scripts/setup_external_libs.sh
scripts/setup_external_libs.sh
```

It performs four tasks:

1. Initializes and updates the pinned submodules:
   - `libs/RDSS`
   - `libs/TabClustPFN`
   - `libs/zcore`
2. Refuses to update a dirty submodule unless
   `ALLOW_DIRTY_EXTERNAL_LIBS=1` is set.
3. Applies local packaging files from `scripts/external_libs/`.
4. Installs the three libraries with:

```bash
python -m pip install --no-build-isolation <library_path>
```


## Local Packaging Files

The files in `external_libs/` provide minimal `pyproject.toml` metadata so the
submodules can be imported by the LUCoS code after editable/local installation.
They are copied into the corresponding submodule directories by the setup
script.

## Notes

- The TabClustPFN checkpoint is still a separate manual download and should be
  placed at `libs/TabClustPFN/checkpoints/step-10000.ckpt`.

