# Publishing the documentation site (GitHub Pages)

The `docs/` folder contains the repository's static documentation landing page.
Keep links repository-relative so a fork or rename does not retain another
user's GitHub account, local filesystem path, or repository name.

## One-time setup

1. Push the repository's default branch to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, choose **Source: Deploy from a branch**.
4. Select the default branch (normally `main`) and **Folder: `/docs`**, then save.
5. GitHub will show the final Pages URL after deployment. For a project site it
   normally has the form `https://<github-user>.github.io/<repository>/`.

Do not hard-code a different contributor's username or an absolute local path in
this file.

## What is published

- `docs/index.html` — static landing page.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architecture, numerical decisions,
  testing strategy and the C++ engineering toolchain.
- The landing page should link back to the repository versions of
  [`../LEARN.md`](../LEARN.md), [`../COOKBOOK.md`](../COOKBOOK.md),
  [`../API_SPEC.md`](../API_SPEC.md) and [`../cpp/README.md`](../cpp/README.md).

Markdown files are primarily intended to be read on GitHub. GitHub Pages may
serve raw Markdown unless a separate static-site generator is configured.

## Keeping it accurate

The site is static, so claims about tests, benchmark coverage or supported build
modes must match executable repository configuration. When test counts, golden
cases, CMake options or CI jobs change, update `README.md`, `docs/index.html` and
the relevant Markdown documentation in the same commit.

Generated CMake metadata is never website content. In particular, files under
`cpp/build*` may contain machine-specific values such as `/Users/<name>/...`;
those directories belong in `.gitignore` and should not be committed or linked.
