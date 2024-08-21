#/usr/bin/ev bash
ROOT=$(dirname $(dirname -- $0))

brew bundle --file="$ROOT"/utils/Brewfile
