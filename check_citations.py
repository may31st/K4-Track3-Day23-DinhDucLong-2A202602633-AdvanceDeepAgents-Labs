"""Standard-library citation validator, uploaded and executed inside the sandbox."""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

REPORT = '/tmp/work/report/report.md'
SOURCES = '/tmp/work/research/sources.json'


def check(report_text, sources):
    """Return problems for malformed sources, unresolved citations and References."""
    if not isinstance(sources, list):
        return ['sources.json must be a JSON list of objects']
    if not sources:
        return ['no sources in sources.json']
    problems, by_n, urls = [], {}, set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f'source at index {index} must be an object')
            continue
        n, url = source.get('n'), source.get('url')
        if type(n) is not int or n < 1:
            problems.append(f'source at index {index} needs a positive integer n')
        elif n in by_n:
            problems.append(f'duplicate source number [{n}]')
        else:
            by_n[n] = source
        try:
            valid_url = isinstance(url, str) and url.startswith(('http://', 'https://')) and bool(urlsplit(url).netloc) and not re.search(r'\s', url)
        except ValueError:
            valid_url = False
        if not valid_url:
            problems.append(f'source [{n}] needs an http(s) URL')
        elif url in urls:
            problems.append(f'duplicate source URL: {url}')
        else:
            urls.add(url)
        family = source.get('source')
        if not isinstance(family, str) or family not in {'arxiv', 'hf-daily', 'hf-search', 'web'}:
            problems.append(f'source [{n}] needs a valid source family string')
            continue
        prefix = {'arxiv': 'https://arxiv.org/abs/',
                  'hf-daily': 'https://huggingface.co/papers/',
                  'hf-search': 'https://huggingface.co/papers/'}.get(family)
        if prefix and (not isinstance(url, str) or not url.startswith(prefix)):
            problems.append(f'source [{n}] URL does not match {family} family; '
                            f'preserve discovery provenance and use {prefix}<id>')

    def mask(match):
        return re.sub(r'[^\n]', ' ', match.group())

    text = re.sub(r'(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$', mask, report_text)
    text = re.sub(r'(`+)[^\n]*?\1', mask, text)
    headings = list(re.finditer(r'(?m)^##[ \t]+References[ \t]*\r?$', text))
    if not headings:
        problems.append('missing ## References heading')
        body, references = text, ''
    else:
        if len(headings) != 1:
            problems.append('report must contain exactly one ## References heading')
        body, references = text[:headings[0].start()], text[headings[0].end():]
    cited = set()
    for match in re.finditer(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\s*\()', body):
        for part in re.split(r'\s*,\s*', match.group(1)):
            span = re.fullmatch(r'(\d+)\s*[–-]\s*(\d+)', part)
            if span:
                start, end = map(int, span.groups())
                if not 0 <= end - start <= 200:
                    problems.append(f'invalid citation range [{part}]')
                    continue
                cited.update(range(start, end + 1))
            elif part.isdigit():
                cited.add(int(part))
            else:
                problems.append(f'invalid citation group [{match.group(1)}]')
    problems.extend(f'[{n}] cited but missing from sources.json' for n in sorted(cited - by_n.keys()))
    problems.extend(f'source [{n}] never cited' for n in sorted(by_n.keys() - cited))
    counts = Counter()
    for line in references.splitlines():
        match = re.match(r'^\[(\d+)\]\s*(.*)$', line)
        if not match:
            if line.strip():
                problems.append('unexpected content in References: ' + line[:100])
            continue
        n = int(match.group(1))
        counts[n] += 1
        found_urls = re.findall(r'https?://[^\s<>]+', match.group(2))
        if n not in by_n:
            problems.append(f'reference [{n}] missing from sources.json')
        if len(found_urls) != 1:
            problems.append(f'reference [{n}] must contain exactly one URL')
        elif n in by_n and found_urls[0] != by_n[n].get('url'):
            problems.append(f'reference [{n}] URL differs from sources.json')
    for n in by_n:
        if counts[n] != 1:
            problems.append(f'source [{n}] needs exactly one reference line (found {counts[n]})')
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding='utf-8') as f:
            report = f.read()
        with open(sources_path, encoding='utf-8') as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f'cannot read inputs: {exc}')
        return 1
    problems = check(report, sources)
    if problems:
        print('\n'.join(problems))
        return 1
    print(f'OK: {len(sources)} sources, all citations resolve')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
