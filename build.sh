#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else "Python 3.8 or newer is required")'
chmod +x kopt
