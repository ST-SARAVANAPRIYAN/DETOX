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
echo -e "  3) Start Backend + Frontend + Telegram Bot (All services)"
echo -e "  4) Start Telegram Bot only"
echo -e "  5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

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
        echo -e "${BLUE}🚀 Starting ALL services (Backend + Frontend + Telegram Bot)...${NC}"
        
        # Check if bot token is set
        if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
            echo -e "${YELLOW}⚠️  Loading TELEGRAM_BOT_TOKEN from .env file...${NC}"
            if [ -f ".env" ]; then
                export $(cat .env | grep -v '^#' | xargs)
            else
                echo -e "${YELLOW}⚠️  .env file not found. Bot token should be in environment.${NC}"
            fi
        fi
        
        # Start backend
        source venv/bin/activate
        cd backend
        python app.py > ../backend.log 2>&1 &
        BACKEND_PID=$!
        cd ..
        
        echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
        sleep 3
        
        # Start Telegram bot
        echo -e "${BLUE}🤖 Starting Telegram Bot (@haki_filter_bot)...${NC}"
        source venv/bin/activate
        python backend/telegram_bot.py > telegram_bot.log 2>&1 &
        BOT_PID=$!
        
        echo -e "${GREEN}✅ Telegram Bot started (PID: $BOT_PID)${NC}"
        sleep 2
        
        # Start frontend
        cd frontend
        npm run dev > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ..
        
        echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
        echo ""
        echo "=================================================="
        echo -e "${GREEN}✅ ALL SERVICES RUNNING!${NC}"
        echo "=================================================="
        echo ""
        echo "📡 Backend API:      http://localhost:5000"
        echo "🤖 Telegram Bot:     @haki_filter_bot"
        echo "🌐 Frontend:         http://localhost:5173"
        echo "📊 Spark UI:         http://localhost:4040"
        echo ""
        echo "Logs:"
        echo "  Backend:       backend.log"
        echo "  Frontend:      frontend.log"
        echo "  Telegram Bot:  telegram_bot.log"
        echo ""
        echo -e "${YELLOW}💡 Test the Telegram bot:${NC}"
        echo "  1. Open Telegram app"
        echo "  2. Search: @haki_filter_bot"
        echo "  3. Send: /start"
        echo "  4. Send any message to analyze"
        echo ""
        echo -e "${YELLOW}🌐 View live feed:${NC}"
        echo "  Open: http://localhost:5173"
        echo "  Go to: Live Detection → Telegram Live Feed tab"
        echo ""
        echo -e "${YELLOW}To stop all servers:${NC}"
        echo "  kill $BACKEND_PID $BOT_PID $FRONTEND_PID"
        echo ""
        echo "=================================================="
        ;;
    4)
        echo ""
        echo -e "${BLUE}🤖 Starting Telegram Bot only...${NC}"
        
        # Check if bot token is set
        if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
            echo -e "${YELLOW}⚠️  Loading TELEGRAM_BOT_TOKEN from .env file...${NC}"
            if [ -f ".env" ]; then
                export $(cat .env | grep -v '^#' | xargs)
            else
                echo -e "${YELLOW}❌ .env file not found!${NC}"
                echo "Create .env file with: TELEGRAM_BOT_TOKEN=your-token-here"
                exit 1
            fi
        fi
        
        echo -e "${GREEN}✅ Bot token loaded${NC}"
        source venv/bin/activate
        python backend/telegram_bot.py
        ;;
    5)
