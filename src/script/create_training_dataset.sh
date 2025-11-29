#!/bin/bash
set -e

# Display current config settings
echo "========================================"
echo "Current dataset configuration:"
echo "========================================"
echo ""
grep -A 7 "^dataset:" src/config/config.yaml
echo ""
echo "========================================"
echo "Is this correct? [y/N]"
read -r answer

if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Please edit src/config/config.yaml and run this script again."
    exit 1
fi

echo "Starting dataset creation..."

uv run python src/dataset/add_edge_to_holicity.py

uv run python src/dataset/create_training_dataset.py

echo "Done!"
