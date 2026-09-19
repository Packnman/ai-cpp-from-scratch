#!/bin/sh
set -eu

revision=5e525c2132c0e87cce890b8b92639a0cf357c1f3
sha256=5796effb85c473aec5b85784545349ed42e408bb0fecdb78f82bc107d53edbe3
output=${1:-data/ner/source/ner.json}
mkdir -p "$(dirname -- "$output")"
url="https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/$revision/ner.json"
temporary="$output.tmp"
curl -L --fail --retry 3 "$url" -o "$temporary"
actual=$(sha256sum "$temporary" | awk '{print $1}')
if [ "$actual" != "$sha256" ]; then
    rm -f "$temporary"
    echo "checksum mismatch: expected $sha256, got $actual" >&2
    exit 1
fi
mv "$temporary" "$output"
printf '%s\n' "source=$url" "revision=$revision" "sha256=$sha256" > "$(dirname -- "$output")/SOURCE"
echo "verified $output ($sha256)"
