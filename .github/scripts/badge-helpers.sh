#!/usr/bin/env bash
# Same badge/report contract as Barograph; reject missing or invalid scores.

write_error_badge() {
  local directory=$1 name=$2 label=$3
  mkdir -p "$directory"
  jq -n --arg label "$label" \
    '{schemaVersion:1, label:$label, message:"error", color:"red", style:"flat"}' \
    > "$directory/$name.json"
}

require_valid_result() {
  local result=$1 directory=$2 name=$3 label=$4
  if ! printf '%s' "$result" | jq -es '
    select(length == 1) | .[0] | select(type == "object") |
    .score | select(type == "number" and . >= 0 and . <= 100)
  ' >/dev/null 2>&1; then
    echo "badge-helpers: missing or invalid .score for $name" >&2
    write_error_badge "$directory" "$name" "$label"
    exit 1
  fi
}

write_badge() {
  local directory=$1 name=$2 label=$3 value=$4 logo_file=${5:-}
  local score color
  score=$(jq -en --arg value "$value" \
    '$value | tonumber | select(. >= 0 and . <= 100)') || return 1
  color=$(jq -rn --argjson score "$score" '
    if $score >= 95 then "brightgreen"
    elif $score >= 80 then "green"
    elif $score >= 60 then "yellow"
    else "red" end
  ') || return 1
  mkdir -p "$directory"
  if [ -n "$logo_file" ] && [ -s "$logo_file" ]; then
    jq -n --arg label "$label" --arg score "$score" --arg color "$color" \
      --rawfile logo "$logo_file" \
      '{schemaVersion:1, label:$label, message:($score + "%"), color:$color, logoSvg:$logo, style:"flat"}' \
      > "$directory/$name.json"
  else
    jq -n --arg label "$label" --arg score "$score" --arg color "$color" \
      '{schemaVersion:1, label:$label, message:($score + "%"), color:$color, style:"flat"}' \
      > "$directory/$name.json"
  fi
}
