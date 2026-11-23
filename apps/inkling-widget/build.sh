#!/bin/zsh
# Build "Inkling Widget.app" (the oMLX desktop widget) and install it in ~/Applications.
# Needs only the Xcode Command Line Tools (swiftc). Usage: build.sh [install-dir]
set -e
HERE=${0:A:h}
DEST=${1:-$HOME/Applications}
APP="$DEST/Inkling Widget.app"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -swift-version 5 -o "$APP/Contents/MacOS/InklingWidget" "$HERE/InklingWidget.swift" -framework AppKit -framework IOKit
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>ai.omlx.inkling-widget</string>
  <key>CFBundleName</key><string>Inkling Widget</string>
  <key>CFBundleExecutable</key><string>InklingWidget</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "Built $APP"
