# Git repository and future commit plan

Repository: `https://github.com/zhuzhenxiang93-create/RAG`

The target directory was initialized on branch `main` after confirming that the remote
was empty. Before the initial commit, `.gitignore` was verified to exclude `.env`,
virtual environments, model weights, uploads, SQLite files, indexes and generated
evaluation/load artifacts. No staged file exceeded 5 MB.

Recommended future commits:

1. `feat(retrieval): integrate production embedding and cross-encoder providers`
2. `test(eval): add independent calibration and held-out evidence labels`
3. `feat(ocr): add local or remote scanned-pdf provider`
4. `perf(index): add persistent vector index and incremental updates`
5. `docs: publish verified full-mode benchmark results`

The completed staged work is represented by one honest initial commit rather than a
fabricated multi-stage history. The repository-local author uses the authenticated
GitHub account's `users.noreply.github.com` address; global Git identity is unchanged.

Before future commits, repeat:

```powershell
git check-ignore .env .venv data\docmind.db artifacts
git status
```
