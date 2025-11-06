#!/bin/bash

# DETOX Web Application - Quick Start
# For CachyOS / Arch-based Linux

clear
echo "=================================================="
echo "   DETOX - Interactive Web Application"
echo "   Quick Start Guide"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}✅ Node.js:${NC} $(node --version)"
echo -e "${GREEN}✅ npm:${NC} $(npm --version)"
echo -e "${GREEN}✅ Python:${NC} $(python --version)"
echo ""
echo "=================================================="
echo ""

echo -e "${BLUE}To start the web application:${NC}"
echo ""
echo -e "${YELLOW}Option 1: Start both servers manually${NC}"
echo ""
echo "  Terminal 1 (Backend):"
echo "    $ source venv/bin/activate"
echo "    $ cd backend"
echo "    $ python app.py"
echo ""
echo "  Terminal 2 (Frontend):"
echo "    $ cd frontend"
echo "    $ npm run dev"
echo ""
echo "=================================================="
echo ""
echo -e "${GREEN}Access URLs:${NC}"
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:5000${NC}"
echo -e "  Spark UI: ${BLUE}http://localhost:4040${NC} (when pipeline runs)"
echo ""
echo "=================================================="
echo ""
echo -e "${YELLOW}Would you like to start the servers now?${NC}"
echo -e "  1) Start Backend only"
echo -e "  2) Start Frontend only"
echo -e "  3) Start both (in background)"
echo -e "  4) Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Starting Flask backend...${NC}"
        source venv/bin/activate
        cd backend
        python app.py
        ;;
    2)
        echo ""
        echo -e "${BLUE}Starting React frontend...${NC}"
        cd frontend
        npm run dev
        ;;
    3)
        echo ""
        echo -e "${BLUE}Starting both servers...${NC}"
        
        # Start backend
        source venv/bin/activate
        cd backend
        python app.py > ../backend.log 2>&1 &
        BACKEND_PID=$!
        cd ..
        
        echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
        sleep 2
        
        # Start frontend
        cd frontend
        npm run dev > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ..
        
        echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
        echo ""
        echo -e "${GREEN}Application is running!${NC}"
        echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
        echo -e "  Backend:  ${BLUE}http://localhost:5000${NC}"
        echo ""
        echo "Logs:"
        echo "  Backend: backend.log"
        echo "  Frontend: frontend.log"
        echo ""
        echo -e "${YELLOW}To stop servers:${NC}"
        echo "  kill $BACKEND_PID $FRONTEND_PID"
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
