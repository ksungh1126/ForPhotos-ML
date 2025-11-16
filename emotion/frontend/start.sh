#!/bin/bash

# ForPhotos 프론트엔드 실행 스크립트

echo "================================================"
echo "  ForPhotos Frontend Server"
echo "================================================"
echo ""

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# 백엔드 서버 확인
echo "🔍 Checking backend server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend server is running at http://localhost:8000"
else
    echo "⚠️  Backend server is not running!"
    echo ""
    echo "Please start the backend server first:"
    echo "  cd /home/work/wonjun/ForPhotos-ML/emotion"
    echo "  uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting frontend server on http://localhost:3000"
echo ""
echo "Open your browser and navigate to:"
echo "  👉 http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Python 3 HTTP 서버 실행
python3 -m http.server 3000
