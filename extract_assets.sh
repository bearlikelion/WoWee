#!/bin/bash
set -euo pipefail

# Usage: ./extract_assets.sh /path/to/WoW/Data [expansion]
#
# Extracts WoW MPQ archives into the Data/ directory for use with wowee.
#
# Arguments:
#   $1  Path to WoW's Data directory (containing .MPQ files)
#   $2  Expansion hint: classic, turtle, tbc, wotlk (optional, auto-detected if omitted)
#
# Examples:
#   ./extract_assets.sh /mnt/games/WoW-3.3.5a/Data
#   ./extract_assets.sh /mnt/games/WoW-1.12/Data classic
#   ./extract_assets.sh /mnt/games/TurtleWoW/Data turtle
#   ./extract_assets.sh /mnt/games/WoW-2.4.3/Data tbc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/bin/asset_extract"
OUTPUT_DIR="${SCRIPT_DIR}/Data"

has_stormlib() {
    # 1) pkg-config names used across distros/packages
    if command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists stormlib 2>/dev/null || pkg-config --exists libstorm 2>/dev/null; then
            return 0
        fi
    fi

    # 2) Direct compile/link probe (catches AUR/system installs without pkg-config metadata)
    if command -v c++ >/dev/null 2>&1; then
        if printf '#include <StormLib.h>\nint main(){return 0;}\n' \
            | c++ -x c++ - -o /dev/null -lStorm -lstdc++ >/dev/null 2>&1; then
            return 0
        fi
        if printf '#include <StormLib.h>\nint main(){return 0;}\n' \
            | c++ -x c++ - -o /dev/null -lstorm >/dev/null 2>&1; then
            return 0
        fi
    fi

    # 3) Fallback checks for typical install locations
    if [ -f /usr/include/StormLib.h ] || [ -f /usr/local/include/StormLib.h ]; then
        if [ -e /usr/lib/libstorm.so ] || [ -e /usr/lib/libStorm.so ] || [ -e /usr/local/lib/libstorm.so ] || [ -e /usr/local/lib/libStorm.so ]; then
            return 0
        fi
    fi

    return 1
}

# --- Validate arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/WoW/Data [classic|turtle|tbc|wotlk]"
    echo ""
    echo "Point this at your WoW client's Data directory."
    echo "The expansion is auto-detected if not specified."
    exit 1
fi

MPQ_DIR="$1"
EXPANSION="${2:-auto}"

if [ ! -d "$MPQ_DIR" ]; then
    echo "Error: Directory not found: $MPQ_DIR"
    exit 1
fi

# If CSV DBCs are already present, extracting binary DBCs is optional.
# Users still need to extract visual assets from their own MPQs.
if [ -d "${OUTPUT_DIR}/expansions" ]; then
    if [ "$EXPANSION" != "auto" ] && [ -d "${OUTPUT_DIR}/expansions/${EXPANSION}/db" ]; then
        if ls "${OUTPUT_DIR}/expansions/${EXPANSION}/db"/*.csv >/dev/null 2>&1; then
            echo "Note: Found CSV DBCs in ${OUTPUT_DIR}/expansions/${EXPANSION}/db/"
            echo "      DBC extraction is optional; visual assets are still required."
            echo ""
        fi
    elif ls "${OUTPUT_DIR}"/expansions/*/db/*.csv >/dev/null 2>&1; then
        echo "Note: Found CSV DBCs under ${OUTPUT_DIR}/expansions/*/db/"
        echo "      DBC extraction is optional; visual assets are still required."
        echo ""
    fi
fi

# Quick sanity check: look for any .MPQ files.
# Use find instead of ls+globs so this works with pipefail and mixed case.
if ! find "$MPQ_DIR" -maxdepth 2 -type f -iname '*.mpq' -print -quit | grep -q .; then
    echo "Error: No .MPQ files found in: $MPQ_DIR"
    echo "Make sure this is the WoW Data/ directory (not the WoW root)."
    exit 1
fi

# --- Build asset_extract if needed ---
if [ ! -f "$BINARY" ]; then
    # --- Check for StormLib (only required to build) ---
    if ! has_stormlib; then
        echo "Error: StormLib not found."
        echo "Install it with: sudo apt install libstorm-dev"
        echo "  or on Arch: sudo pacman -S stormlib"
        echo "  or build from source: https://github.com/ladislav-zezula/StormLib"
        exit 1
    fi

    echo "Building asset_extract..."
    # Always configure first so CMake can (re)detect optional deps like StormLib.
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target asset_extract -- -j"$(nproc)"
    echo ""
fi

if [ ! -f "$BINARY" ]; then
    echo "Error: Failed to build asset_extract"
    exit 1
fi

# --- Run extraction ---
echo "Extracting assets from: $MPQ_DIR"
echo "Output directory:       $OUTPUT_DIR"
echo ""

EXTRA_ARGS=()
if [ "$EXPANSION" != "auto" ]; then
    EXTRA_ARGS+=(--expansion "$EXPANSION")
fi

if [ "$EXPANSION" != "auto" ] && [ -d "${OUTPUT_DIR}/expansions/${EXPANSION}/db" ] && ls "${OUTPUT_DIR}/expansions/${EXPANSION}/db"/*.csv >/dev/null 2>&1; then
    EXTRA_ARGS+=(--skip-dbc)
elif ls "${OUTPUT_DIR}"/expansions/*/db/*.csv >/dev/null 2>&1; then
    EXTRA_ARGS+=(--skip-dbc)
fi

"$BINARY" --mpq-dir "$MPQ_DIR" --output "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"

echo ""
echo "Done! Assets extracted to $OUTPUT_DIR"
echo "You can now run wowee."
