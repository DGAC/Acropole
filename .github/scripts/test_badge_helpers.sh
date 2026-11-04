#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/badge-helpers.sh"
badge_test_dir=$(mktemp -d)
trap 'rm -rf "$badge_test_dir"' EXIT

for payload in '{"score":0}' '{"score":97.5}' '{"score":100}'; do
  (require_valid_result "$payload" "$badge_test_dir" audit audit)
done
for payload in '' '{}' '{"score":null}' '{"score":"97"}' '{"score":-1}' '{"score":101}' 'invalid' '{"score":97} {"score":98}'; do
  if (require_valid_result "$payload" "$badge_test_dir" audit audit) 2>/dev/null; then
    echo "Accepted invalid report: $payload" >&2
    exit 1
  fi
  jq -e '.message == "error" and .color == "red"' "$badge_test_dir/audit.json" >/dev/null
done

for pair in '0:red' '60:yellow' '80:green' '97.5:brightgreen' '100:brightgreen'; do
  score=${pair%:*}
  color=${pair#*:}
  write_badge "$badge_test_dir" audit audit "$score"
  jq -e --arg message "$score%" --arg color "$color" \
    '.schemaVersion == 1 and .message == $message and .color == $color and (has("logoSvg") | not)' \
    "$badge_test_dir/audit.json" >/dev/null
done
for score in '' null -1 101; do
  if write_badge "$badge_test_dir" invalid audit "$score" 2>/dev/null; then
    echo "Accepted invalid badge score: $score" >&2
    exit 1
  fi
done
printf '<svg xmlns="http://www.w3.org/2000/svg"></svg>' > "$badge_test_dir/logo.svg"
write_badge "$badge_test_dir" audit audit 97 "$badge_test_dir/logo.svg"
jq -e '.logoSvg | startswith("<svg")' "$badge_test_dir/audit.json" >/dev/null
touch "$badge_test_dir/empty.svg"
write_badge "$badge_test_dir" audit audit 97 "$badge_test_dir/empty.svg"
jq -e 'has("logoSvg") | not' "$badge_test_dir/audit.json" >/dev/null
echo "Badge report validation, percentages, colors and logos: passed"
