from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth and data.strip():
            self.values.append(data.strip())

    def text(self) -> str:
        return "\n".join(self.values)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://www.stats.gov.cn/gk/tjfg/xgfxfg/202503/t20250311_1958931.html",
    )
    parser.add_argument("--output-dir", default="data/raw/statutes")
    args = parser.parse_args()
    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": "LegalMind-RAG research data audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        content_type = response.headers.get_content_charset() or "utf-8"
    html = raw.decode(content_type, errors="replace")
    extractor = VisibleTextExtractor()
    extractor.feed(html)
    text = re.sub(r"[ \t]+", " ", extractor.text())
    article_count = len(re.findall(r"第[一二三四五六七八九十百千零〇0-9]+条", text))
    if article_count < 400:
        raise ValueError(f"Downloaded page yielded only {article_count} article markers")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "criminal_law_official.html"
    text_path = output / "criminal_law_official.txt"
    html_path.write_bytes(raw)
    text_path.write_text(text + "\n", encoding="utf-8")
    manifest = {
        "title": "中华人民共和国刑法",
        "source_url": args.url,
        "source_organization": "国家统计局",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": sha256(raw),
        "text_sha256": sha256((text + "\n").encode("utf-8")),
        "article_markers": article_count,
        "source_status": "official_government_url_automatically_downloaded_unreviewed",
        "version_note": "page states Criminal Law as amended through Amendment XII; verify before legal use",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
