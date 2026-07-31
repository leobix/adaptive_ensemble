# GitHub release checklist

## Before the first commit

```bash
python scripts/ci/repository_audit.py .
julia --startup-file=no --project=. --threads=4 test/runtests.jl
```

When a Gurobi license is available, also run the Windows setup script or set
`RUN_GUROBI_TESTS=1` before the direct Julia test command.

## Large archival files

Two CSV files exceed 50 MiB. They can be pushed by Git on the command line because they remain below
100 MiB, but GitHub will warn and browser upload is not suitable. Before a public first commit,
consider moving them to Git LFS or a release/data archive after checking redistribution rights:

```bash
git lfs install
git lfs track data/X_test.csv data/energy_X_past6_future1.csv
git add .gitattributes
```

Do this before the first public commit; retrofitting LFS later requires history migration. This source
archive intentionally keeps ordinary full files and does not require Git LFS.

## Create the repository commit

```bash
git init
git branch -M main
git add --all
git diff --cached --check
git status --short
git commit -m "Release corrected Julia publication pipeline"
```

Then connect the desired GitHub repository and push:

```bash
git remote add origin <repository-url>
git push -u origin main
```

## After pushing

- Confirm the Julia CI workflow passes.
- Enable branch protection and require the CI check.
- Enable GitHub secret scanning when available.
- Review `DATA_AND_LICENSES.md` before changing repository visibility to public.
- Publish generated experiment artifacts separately; do not commit `publication_artifacts*`.
