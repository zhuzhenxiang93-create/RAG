# Git repository and future commit plan

Repository: `https://github.com/zhuzhenxiang93-create/RAG`

The target directory was initialized on branch `main` after confirming that the remote
was empty. Before the initial commit, `.gitignore` was verified to exclude `.env`,
virtual environments, model weights, uploads, SQLite files, indexes and generated
evaluation/load artifacts. No staged file exceeded 5 MB.

Recommended future commits:

1. `feat(intent): add MASSIVE LoRA classification and adaptive RAG routing`
2. `test(intent): add classification metrics and route integration coverage`
3. `docs(intent): document reproducible training and honest evaluation boundaries`
4. `feat(retrieval): integrate production embedding and cross-encoder providers`
5. `test(eval): add independent calibration and held-out evidence labels`

For the current intent-routing change, the first three commits are a recommended split.
Do not commit generated MASSIVE data, checkpoints, `.env` files or evaluation artifacts.

The completed staged work is represented by one honest initial commit rather than a
fabricated multi-stage history. The repository-local author uses the authenticated
GitHub account's `users.noreply.github.com` address; global Git identity is unchanged.

Before future commits, repeat:

```powershell
git check-ignore .env .venv data\docmind.db artifacts
git status
```
