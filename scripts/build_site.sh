#!/usr/bin/env bash
# Assemble the static site (simulation + figures + report PDF) into _site/.
#   bash scripts/build_site.sh && python -m http.server -d _site 8000
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site
mkdir -p _site/report _site/figures
cp -r web/. _site/
rm -f _site/js/crosscheck.mjs
cp report/report.pdf _site/report/
cp figures/*.png _site/figures/
echo "site assembled in _site/"
