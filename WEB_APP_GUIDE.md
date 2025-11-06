# DETOX - Interactive Web Application Setup Guide

## 🎉 Web Application Complete!

Your PySpark toxicity detection project now has a beautiful, interactive web interface!

## 📦 What's New

### Backend (Flask API)
- ✅ REST API with 7 endpoints
- ✅ WebSocket support for real-time updates
- ✅ Pipeline orchestration
- ✅ Step-by-step execution control

### Frontend (React + Vite)
- ✅ Modern gradient UI design
- ✅ Welcome page with project overview
- ✅ Interactive demo with step navigation
- ✅ Real-time terminal output viewer
- ✅ Embedded Spark Web UI monitor
- ✅ Results visualization with charts
- ✅ Smooth animations and transitions

## 🚀 How to Run the Web Application

### Step 1: Install Node.js (if not installed)

#### On Ubuntu/Debian:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### On macOS:
```bash
brew install node
```

#### On Windows:
Download from: https://nodejs.org/

### Step 2: Install Frontend Dependencies
```bash
cd /home/saravana/projects/ssfproject/frontend
npm install
```

### Step 3: Start the Backend (Flask)
Open Terminal 1:
```bash
cd /home/saravana/projects/ssfproject
source venv/bin/activate
cd backend
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Spark UI will be available at http://localhost:4040 when running
```

### Step 4: Start the Frontend (React)
Open Terminal 2:
```bash
cd /home/saravana/projects/ssfproject/frontend
npm run dev
```

You should see:
```
  VITE v5.0.8  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 5: Access the Application
Open your browser and go to: **http://localhost:5173**

## 🎨 Application Structure

### Welcome Page (/)
- Project overview and features
- Key statistics
- Tech stack showcase
- "Start Interactive Demo" button

### Demo Page (/demo)

#### Sidebar (Left)
- All 9 pipeline steps listed
- Color-coded status indicators:
  - ⚪ Pending (not started)
  - 🔵 Running (in progress)
  - ✅ Completed (successful)
  - ❌ Failed (error)
- Progress bar at bottom

#### Main Content (Right)
**Tab 1: Step Details**
- Current step explanation
- What the step does
- Technical implementation details
- Code snippets
- Estimated execution time

**Tab 2: Terminal Output**
- Real-time command execution logs
- Syntax-highlighted output
- Auto-scroll to latest
- Beautiful terminal theme

**Tab 3: Spark Monitor**
- Embedded Spark Web UI (http://localhost:4040)
- Jobs, Stages, Storage, Executors
- Real-time Spark metrics
- Option to open in new tab

**Tab 4: Results** (appears after completion)
- Model performance metrics
- User toxicity distribution pie chart
- Performance bar chart
- Sample predictions table
- Download CSV buttons
- Execution statistics

#### Header Controls
- **Previous**: Go to previous step
- **Run Step**: Execute current step
- **Next**: Go to next step
- **Run All**: Execute entire pipeline

## 📁 File Structure

```
ssfproject/
├── backend/
│   └── app.py                 # Flask API (442 lines)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── StepViewer.jsx        # Step details component
│   │   │   ├── StepViewer.css
│   │   │   ├── TerminalOutput.jsx    # Terminal viewer
│   │   │   ├── TerminalOutput.css
│   │   │   ├── SparkMonitor.jsx      # Spark UI embed
│   │   │   ├── SparkMonitor.css
│   │   │   ├── ResultsView.jsx       # Charts & results
│   │   │   └── ResultsView.css
│   │   ├── pages/
│   │   │   ├── Welcome.jsx           # Landing page
│   │   │   ├── Welcome.css
│   │   │   ├── Demo.jsx              # Main demo page
│   │   │   └── Demo.css
│   │   ├── App.jsx                   # Router setup
│   │   ├── App.css
│   │   ├── index.css                 # Global styles
│   │   └── main.jsx                  # Entry point
│   ├── index.html
│   ├── package.json                  # Dependencies
│   └── vite.config.js                # Vite config
└── ... (existing PySpark files)
```

## 🎯 API Endpoints

The Flask backend provides these endpoints:

### REST Endpoints
1. **GET /api/project-info**
   - Returns project metadata
   - Response: `{ name, description, version, tech_stack }`

2. **GET /api/pipeline-steps**
   - Returns all 9 pipeline steps
   - Response: `{ steps: [...] }`

3. **POST /api/execute-step/<step_id>**
   - Executes specific step (1-9)
   - Response: `{ success, message }`

4. **POST /api/execute-all**
   - Executes entire pipeline
   - Response: `{ success, message }`

5. **GET /api/results**
   - Returns execution results
   - Response: `{ metrics, predictions, user_analysis }`

6. **GET /api/sample-predictions**
   - Returns sample predictions
   - Response: `{ predictions: [...] }`

7. **GET /api/spark-status**
   - Checks Spark session status
   - Response: `{ active, ui_url }`

### WebSocket Events
- **connect**: Client connected
- **disconnect**: Client disconnected
- **step_started**: Emitted when step starts
- **step_progress**: Progress updates during execution
- **step_completed**: Step finished (success/fail)
- **pipeline_completed**: All steps completed

## 🎨 UI Features

### Animations
- Smooth page transitions (Framer Motion)
- Hover effects on cards
- Loading spinners
- Progress bars
- Cursor blinking in terminal

### Responsive Design
- Desktop: 1400px max width
- Tablet: Adaptive layout
- Mobile: Stacked views

### Color Scheme
- Primary: #667eea (Purple Blue)
- Secondary: #764ba2 (Purple)
- Accent: #f093fb (Pink)
- Success: #10b981 (Green)
- Warning: #f59e0b (Orange)
- Error: #ef4444 (Red)

### Fonts
- Main: Inter (Google Fonts)
- Code: Courier New (monospace)

## 🔧 Customization

### Change Colors
Edit `frontend/src/index.css`:
```css
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  /* ... other colors */
}
```

### Change API Port
Edit `frontend/vite.config.js`:
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:5000', // Change this
    }
  }
}
```

Edit `backend/app.py`:
```python
socketio.run(app, host='0.0.0.0', port=5000, debug=True)  # Change port
```

### Add More Steps
Edit `backend/app.py` and add to `PIPELINE_STEPS` array:
```python
{
    'id': 10,
    'name': 'Your New Step',
    'description': 'What this step does',
    'status': 'pending'
}
```

## 📊 Dependencies

### Python (requirements.txt)
- pyspark==3.5.3
- pandas>=2.2.0
- flask==3.0.0
- flask-cors==4.0.0
- flask-socketio==5.3.5
- python-socketio==5.10.0
- eventlet==0.35.2

### Node.js (package.json)
- react: ^18.2.0
- react-dom: ^18.2.0
- react-router-dom: ^6.20.0
- axios: ^1.6.2
- socket.io-client: ^4.5.4
- framer-motion: ^10.16.16
- recharts: ^2.10.3
- lucide-react: ^0.294.0
- vite: ^5.0.8

## 🐛 Troubleshooting

### "npm: command not found"
**Solution**: Install Node.js first (see Step 1 above)

### Port 5173 already in use
**Solution**: Kill the process or change the port in `vite.config.js`

### Port 5000 already in use
**Solution**: Kill the process or change Flask port in `app.py`

### Spark UI not loading in iframe
**Solution**: 
- Run any pipeline step to start Spark session
- Click "Open in New Tab" button
- Check http://localhost:4040 directly

### CORS errors
**Solution**: Ensure Flask-CORS is installed and configured in `app.py`

### WebSocket connection failed
**Solution**: 
- Check Flask server is running
- Verify port 5000 is accessible
- Check browser console for errors

## 🚀 Next Steps

### Development
1. Install Node.js and npm
2. Run `npm install` in frontend directory
3. Start both servers (backend + frontend)
4. Test the application
5. Make any customizations

### Production Deployment
1. Build frontend: `npm run build`
2. Serve static files from Flask
3. Use production WSGI server (Gunicorn)
4. Set up reverse proxy (Nginx)
5. Configure SSL certificates

### Enhancements
- [ ] Add user authentication
- [ ] Implement result caching
- [ ] Add export to PDF
- [ ] Create admin dashboard
- [ ] Add A/B testing for models
- [ ] Implement batch processing
- [ ] Add email notifications

## 📚 Resources

- [React Documentation](https://react.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Vite Documentation](https://vitejs.dev/)
- [Socket.IO Documentation](https://socket.io/)

## 🎉 Success!

Your interactive web application is ready! Enjoy exploring the toxicity detection pipeline with a beautiful UI! 🚀

**Made with ❤️ using PySpark, Flask, and React**
