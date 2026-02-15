#!/bin/bash
# Simple wrapper script for generating PDF reports
# Usage: ./generate_report.sh [folder_name] [output_name]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
FOLDER="${1:-comparison_results}"
OUTPUT="${2:-${FOLDER}/benchmark_report.pdf}"

echo -e "${BLUE}📊 Benchmark PDF Report Generator${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""
echo -e "Folder: ${GREEN}${FOLDER}${NC}"
echo -e "Output: ${GREEN}${OUTPUT}${NC}"
echo ""

# Check if folder exists
if [ ! -d "$FOLDER" ]; then
    echo -e "${YELLOW}⚠️  Warning: Folder '${FOLDER}' does not exist${NC}"
    echo ""
    echo "Usage: $0 [folder_name] [output_name]"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default (comparison_results)"
    echo "  $0 my_results                         # Use custom folder"
    echo "  $0 my_results reports/my_report.pdf   # Custom folder and output"
    echo ""
    exit 1
fi

# Generate the report
python3 ../src/generate_pdf_report.py --folder "$FOLDER" --output "$OUTPUT"

echo ""
echo -e "${GREEN}✅ Done!${NC}"
echo -e "Report saved to: ${GREEN}${OUTPUT}${NC}"

# Ask if user wants to open the report
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open the report now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT"
    fi
fi
