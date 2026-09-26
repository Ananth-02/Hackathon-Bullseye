import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "viewer_template.html"


def build_viewer(datasets, out):
    payload = json.dumps(datasets).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__BULLSEYE_DATA__*/[]", payload)
    Path(out).write_text(html)
