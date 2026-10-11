#!/usr/bin/env python
"""Refreshes the file of Cloudflare proxy ranges that the server trusts for CF-Connecting-IP.

    python tools/refresh_cloudflare_ips.py [path]      (default: the file in the server's source folder; give the path of the DEPLOYED file on the server)

Downloads https://www.cloudflare.com/ips-v4 and /ips-v6, checks that every line is a CIDR, and writes the file atomically. The running server reads
the file at start: restart the service after a refresh. Cloudflare publishes changes in advance and rarely; check a few times a year.
"""
import ipaddress
import os
import sys
import tempfile
import urllib.request
from datetime import date

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csharp', 'src', 'TTC.Siren.Server', 'data', 'cloudflare-ips.txt')


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return [l.strip() for l in r.read().decode('ascii').splitlines() if l.strip()]


def main():
    path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    ranges = fetch('https://www.cloudflare.com/ips-v4') + fetch('https://www.cloudflare.com/ips-v6')
    for r in ranges:
        ipaddress.ip_network(r, strict=True)  # raises on anything that is not a CIDR
    if len(ranges) < 10:
        raise SystemExit('refusing to write: only %d ranges were returned' % len(ranges))
    body = ('# Cloudflare\'s published proxy ranges (https://www.cloudflare.com/ips-v4 and https://www.cloudflare.com/ips-v6), fetched %s.\n'
            '# One CIDR per line; lines starting with # are comments. Refresh with: python tools/refresh_cloudflare_ips.py [path] (see docs/DEPLOY_WINDOWS.md, Cloudflare).\n'
            % date.today().isoformat()) + '\n'.join(ranges) + '\n'
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    with os.fdopen(fd, 'w', newline='\n') as f:
        f.write(body)
    os.replace(tmp, path)
    print('wrote %d ranges to %s' % (len(ranges), path))


if __name__ == '__main__':
    main()
