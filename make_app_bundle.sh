#!/usr/bin/env bash
#
# Build "VisoMaster Fusion.app" so the macOS dock shows the real icon and name.
#
# Why this is needed at all: launched as `python main.py`, macOS shows a generic
# interpreter glyph labelled "Python" in the dock. The dock icon comes from the
# application *bundle*, not from Qt's window icon -- setApplicationDisplayName()
# fixes the label and the task switcher, but only a bundle with a CFBundleIcon
# replaces the image itself.
#
# The bundle's executable is a shell stub rather than the interpreter, so the
# existing .venv is used as-is and nothing needs relocating.
#
# Usage:  ./make_app_bundle.sh [destination]
# Default destination is ~/Applications.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/VisoMaster Fusion.app"
PY="$REPO/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "error: $PY not found. The bundle runs the repo's virtualenv." >&2
    exit 1
fi

echo "==> generating icon"
ICONSET="$(mktemp -d)"
trap 'rm -rf "$ICONSET"' EXIT

"$PY" - "$REPO/app/ui/core/media/visomaster_small.png" "$ICONSET" <<'PYEOF'
import sys
from PIL import Image

src_path, out_dir = sys.argv[1], sys.argv[2]
src = Image.open(src_path).convert("RGBA")
w, h = src.size
if w != h:                      # pad to square so iconutil's sizes line up
    side = max(w, h)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(src, ((side - w) // 2, (side - h) // 2))
    src = sq

base = src.resize((1024, 1024), Image.LANCZOS)
for px in (16, 32, 64, 128, 256, 512, 1024):
    base.resize((px, px), Image.LANCZOS).save(f"{out_dir}/icon_{px}x{px}.png")
    if px != 1024:
        base.resize((px * 2, px * 2), Image.LANCZOS).save(f"{out_dir}/icon_{px}x{px}@2x.png")
PYEOF

mkdir -p "$ICONSET/VisoMaster.iconset"
# PIL wrote icon_*x*.png into $ICONSET; move them into the .iconset directory
mv "$ICONSET"/icon_*.png "$ICONSET/VisoMaster.iconset/"
iconutil -c icns "$ICONSET/VisoMaster.iconset" -o "$ICONSET/VisoMaster.icns"

echo "==> building $APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ICONSET/VisoMaster.icns" "$APP/Contents/Resources/VisoMaster.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>VisoMaster Fusion</string>
    <key>CFBundleDisplayName</key>
    <string>VisoMaster Fusion</string>
    <key>CFBundleExecutable</key>
    <string>VisoMaster Fusion</string>
    <key>CFBundleIdentifier</key>
    <string>com.visomaster.fusion</string>
    <key>CFBundleIconFile</key>
    <string>VisoMaster</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.video</string>
</dict>
</plist>
PLIST

# The stub chdir's into the repo: the app reads app/ui/styles/*.qss and
# model_assets by relative path, and macOS sets cwd to / for bundle launches.
cat > "$APP/Contents/MacOS/VisoMaster Fusion" <<STUB
#!/bin/sh
cd "$REPO" || exit 1
exec "$REPO/.venv/bin/python" main.py "\$@"
STUB
chmod +x "$APP/Contents/MacOS/VisoMaster Fusion"

# Ad-hoc signature: unsigned bundles get a hard Gatekeeper block on Apple
# Silicon, and an ad-hoc sig at least keeps it launching locally.
codesign --force --deep --sign - "$APP" 2>/dev/null \
    || echo "    (codesign unavailable; the bundle still runs locally)"

touch "$APP"

echo
echo "Built $APP"
echo "Drag it to your Dock, or:  open \"$APP\""
echo "Note: it points at $REPO -- move or delete that and the bundle breaks."
