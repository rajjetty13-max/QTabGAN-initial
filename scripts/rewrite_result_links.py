"""Rewrite local Markdown links in unmoved docs after the results relocation."""
import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote


LINK = re.compile(r'(?P<start>!?\[[^\]]*\]\()(?P<target>[^\)]+)(?P<end>\))')


def rewrite(document, moves, project):
    """Return updated text and moved targets that are unexpectedly absent."""
    text = document.read_text()
    missing = []

    def replace(match):
        target = match.group('target')
        angle = target.startswith('<') and target.endswith('>')
        raw = target[1:-1] if angle else target
        if not raw or raw.startswith('#') or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', raw):
            return match.group(0)
        if re.search(r'\s+[\"\']', raw):
            return match.group(0)
        path, marker, anchor = raw.partition('#')
        decoded = Path(unquote(path))
        old = decoded if decoded.is_absolute() else document.parent / decoded
        old = Path(os.path.normpath(old))
        new = old
        for source, destination in moves:
            try:
                new = destination / old.relative_to(source)
                break
            except ValueError:
                pass
        if new == old:
            return match.group(0)
        target = os.path.relpath(new, document.parent)
        if marker:
            target += '#' + anchor
        if angle:
            target = '<' + target + '>'
        if not new.exists():
            missing.append(str(new.relative_to(project)))
        return match.group('start') + target + match.group('end')

    return LINK.sub(replace, text), missing


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('documents', nargs='+', type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=Path('results/history/relocation_manifest.json'))
    parser.add_argument('--apply', action='store_true', help='Write updates; otherwise dry run')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text())
    moves = sorted([(project / row['from'], project / row['to']) for row in manifest['moves']],
                   key=lambda pair: len(pair[0].parts), reverse=True)
    for document in args.documents:
        document = document.resolve()
        updated, missing = rewrite(document, moves, project)
        changed = updated != document.read_text()
        if changed and args.apply:
            document.write_text(updated)
        print(json.dumps({'document': str(document.relative_to(project)),
                          'changed': changed, 'written': changed and args.apply,
                          'missing_relocated_targets': missing}))
        if missing:
            parser.exit(1, 'A relocated link target is missing.\n')


if __name__ == '__main__':
    main()
