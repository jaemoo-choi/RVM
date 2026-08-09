# Project Page Preview

This branch stores a private static project-page draft. Do not enable GitHub
Pages for this repository before the paper is public, unless the hosting service
is explicitly configured with private access control.

## What To Edit

- Main page: `index.html`.
- Institution logos: `logos/GT.svg` and `logos/NV.svg`.
- Web figures: `figures/`.
- Video metadata: `wan21/prompts.json` and `skyreels/prompts.json`.
- Video files are organized by model/method folders under `wan21/` and
  `skyreels/`; `index.html` reads the JSON metadata and builds the gallery.

Current placeholders:

- Author list follows the current manuscript draft.
- `Paper` and `Code` buttons intentionally have empty links for now; the Code
  button is labeled `Code (coming soon)`.
- BibTeX is still `TBA`.

Math on the page should be rendered with KaTeX. Inline math can use the
`data-katex` attribute in `index.html`; if the CDN is unavailable, the main loss
formula keeps a plain-text fallback.

## Radar Plot

The displayed radar plot is `figures/vbench_t2v_radar_wide.png`. Treat this file as
the canonical web asset inside this project-page repository.

If the paper repository regenerates the radar plot, replace this web asset with
the latest wide PNG output:

```bash
cp /path/to/paper_repo/radar_plot/vbench_t2v_radar_wide.png figures/vbench_t2v_radar_wide.png
```

The source CSV and plotting script live in the paper repository, not necessarily
inside this project-page checkout.

## Local Preview

From this directory:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8000
```

The `--bind 127.0.0.1` flag keeps the server local to your machine. Use the
local HTTP server instead of opening `index.html` directly, since browser
behavior for videos and relative paths is closer to a deployed site over HTTP.

Stop the server with `Ctrl-C`.

## Sync

Before pushing, inspect the staged scope:

```bash
git status
```

For the current project page assets, the intended source files are:

```bash
git add index.html README.md .gitignore logos figures
```

If videos or metadata changed, add the relevant `wan21/` or `skyreels/` files as
well. After staging the intended files, sync the branch:

```bash
git commit -m update
git pull --rebase origin project_page
git push origin project_page
```

The repository branch can remain private on GitHub. A private repository does
not become a public website unless GitHub Pages or another hosting service is
enabled.
