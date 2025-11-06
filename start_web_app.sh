#!/bin/bash

# DETOX Web Application Launcher
# This script helps you start the Flask backend and React frontend

echo "=================================================="
echo "   DETOX - Chat Message Toxicity Detector"
echo "   Web Application Launcher"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Node.js is installed
echo -e "${BLUE}[1/5] Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed!${NC}"
    echo ""
    echo "Please install Node.js first:"
    echo "  Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs"
    echo "  macOS: brew install node"
    echo "  Windows: Download from https://nodejs.org/"
    exit 1
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js is installed: ${NODE_VERSION}${NC}"
fi

# Check if Python virtual environment exists
echo -e "${BLUE}[2/5] Checking Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Virtual environment created and dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Check if frontend dependencies are installed
echo -e "${BLUE}[3/5] Checking frontend dependencies...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend dependencies not installed${NC}"
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Frontend dependencies already installed${NC}"
fi

# Check if Flask dependencies are installed
echo -e "${BLUE}[4/5] Checking backend dependencies...${NC}"
source venv/bin/activate
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Flask not installed${NC}"
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Backend dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Backend dependencies already installed${NC}"
fi

# Start the application
echo ""
echo -e "${BLUE}[5/5] Starting the application...${NC}"
echo ""
echo "=================================================="
echo "   Starting DETOX Web Application"
echo "=================================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Servers stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Start backend
echo -e "${BLUE}Starting Flask backend...${NC}"
source venv/bin/activate
cd backend
python app.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3

if ps -p $BACKEND_PID > /dev/null; then
    echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
    echo -e "   URL: ${BLUE}http://localhost:5000${NC}"
else
    echo -e "${RED}❌ Backend failed to start. Check backend.log${NC}"
    exit 1
fi

# Start frontend
echo -e "${BLUE}Starting React frontend...${NC}"
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 5

if ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
    echo -e "   URL: ${BLUE}http://localhost:5173${NC}"
else
    echo -e "${RED}❌ Frontend failed to start. Check frontend.log${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=================================================="
echo -e "${GREEN}   🎉 Application Started Successfully!${NC}"
echo "=================================================="
echo ""
echo -e "${GREEN}Access the application:${NC}"
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:5000${NC}"
echo -e "  Spark UI: ${BLUE}http://localhost:4040${NC} ${YELLOW}(when pipeline runs)${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
echo ""

# Keep script running
wait
