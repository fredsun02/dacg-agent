#!/bin/bash
# Run KGSA Web Interface

cd "$(dirname "$0")"

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

echo "Starting KGSA Web Interface..."
echo "Open http://localhost:5000 in your browser"
echo ""

python app.py
