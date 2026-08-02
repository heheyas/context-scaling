# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Assemble static/index.html from template + css + js.

The static SPA is shipped as three separate files (template.html,
style.css, script.js) for easy diffing during development. This script
inlines the CSS and JS into the template's %%CSS%% and %%SCRIPT%%
placeholders and writes the combined single-file HTML that FastAPI
serves from GET /.

Run once after any change to demo/static/{template.html,style.css,script.js}:

    python demo/build_index.py
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


def build(output_path: Path) -> None:
    template = (STATIC / "template.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    js = (STATIC / "script.js").read_text(encoding="utf-8")

    html = template
    if "%%CSS%%" in html:
        html = html.replace("%%CSS%%", css)
    else:
        # Fallback: inject just before </head>.
        html = html.replace("</head>", f"<style>\n{css}\n</style>\n</head>", 1)
    if "%%SCRIPT%%" in html:
        html = html.replace("%%SCRIPT%%", js)
    else:
        # Fallback: inject just before </body>.
        html = html.replace("</body>", f"<script>\n{js}\n</script>\n</body>", 1)
    html = html.replace("%%GENERATION_TIME%%", datetime.utcnow().isoformat() + "Z")

    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(html):,} chars)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=STATIC / "index.html")
    args = ap.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
