# FedLearnDR Web Application - Complete Instructions

## 📁 Project Structure

```
project/
├── federated-dr/          # Original ML project (Python)
│   ├── src/               # Federated learning code
│   ├── experiments/       # Training experiments
│   ├── configs/           # Configuration files
│   └── ...
│
└── web/                   # NEW: Web application
    ├── src/               # React frontend
    │   ├── components/    # All UI components
    │   ├── App.jsx        # Main app
    │   └── index.css      # Global styles
    ├── server/            # Express backend
    │   ├── index.js       # API server
    │   └── uploads/       # Temp file storage
    ├── package.json       # Frontend dependencies
    └── GETTING_STARTED.md # Quick start guide
```

## 🚀 Quick Start

### 1. Navigate to Web Directory
```bash
cd web
```

### 2. Install Dependencies

**Frontend:**
```bash
npm install
```

**Backend:**
```bash
cd server
npm install
cd ..
```

### 3. Start the Application

**Option A: Two Terminals (Recommended)**

Terminal 1 - Start Backend:
```bash
cd server
npm start
```
✅ Backend running on http://localhost:3000

Terminal 2 - Start Frontend:
```bash
# From web/ directory
npm run dev
```
✅ Frontend running on http://localhost:5173

**Option B: Single Command (Unix/Linux/Mac)**
```bash
chmod +x start.sh
./start.sh
```

### 4. Open in Browser
Navigate to: **http://localhost:5173**

## 🎨 Features

### Landing Page (Hero)
- Animated gradient background
- 4 key metrics cards
- Call-to-action buttons
- Smooth scroll indicators

### How It Works
- 4-step visual guide
- Federated vs Traditional ML comparison
- Animated step cards

### Live Demo ⭐
- **Drag & drop** image upload
- Hospital selection dropdown
- Real-time DR analysis
- Beautiful result cards with:
  - DR severity (0-4 grades)
  - Confidence meter
  - Risk level badges
  - Clinical findings
  - Hospital info

### Results Dashboard
- Training curves (Line chart)
- Accuracy comparison (Bar chart)
- Per-hospital performance (Radar chart)
- 4 metric summary cards

### About Project
- Problem statement
- Solution overview
- Technical approach
- Animated statistics

### Technical Details
- Collapsible section
- Model architecture specs
- Federated learning details
- Code snippets

### FAQ
- 5 common questions
- Accordion-style UI
- Smooth animations

### Footer
- Navigation links
- GitHub repository link
- Tech stack info
- Social media

## 🎯 Using the Demo

1. **Upload Image:**
   - Click or drag/drop a retinal fundus image
   - Supported: PNG, JPG, JPEG (max 5MB)

2. **Select Hospital:**
   - Choose from 4 hospitals:
     - Metropolitan General Hospital
     - Vision Care Specialized Center
     - Urban Health Medical Center
     - Advanced Research Institute

3. **Analyze:**
   - Click "Analyze Image" button
   - Wait for processing (~1.5 seconds)

4. **View Results:**
   - DR severity grade (0-4)
   - Confidence score
   - Risk level (low/medium/high/critical)
   - Clinical findings
   - Hospital and analysis info

## 🎨 Design Features

### Color Scheme
- Primary Blue: `#2563eb`
- Primary Green: `#10b981`
- Medical aesthetic throughout

### Dark Mode
- Toggle in top-right corner
- Persisted in localStorage
- Smooth transitions

### Responsive Design
- Mobile: 320px+
- Tablet: 768px+
- Desktop: 1024px+

### Animations
- Framer Motion throughout
- Smooth page transitions
- Interactive hover effects
- Loading states

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI library
- **Vite** - Build tool & dev server
- **Framer Motion** - Animation library
- **Chart.js** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icon library

### Backend
- **Express.js** - Web framework
- **Multer** - File upload handling
- **CORS** - Cross-origin support
- **Node.js** - Runtime

### Styling
- Custom CSS with CSS Variables
- Mobile-first approach
- No CSS frameworks
- Dark mode support

## 📡 API Reference

### Endpoints

**POST** `/api/upload`
- Upload retinal image
- Max size: 5MB
- Returns: `{ imageId, preview_url }`

**POST** `/api/analyze`
- Analyze uploaded image
- Body: `{ imageId, hospitalId }`
- Returns: DR analysis results

**GET** `/api/hospitals`
- Get list of hospitals
- Returns: Array of hospital objects

**GET** `/api/metrics`
- Get dashboard metrics
- Returns: Charts data and statistics

## 🔧 Configuration

### Port Configuration

**Frontend** (default: 5173)
Edit `vite.config.js`:
```javascript
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
})
```

**Backend** (default: 3000)
Edit `server/index.js`:
```javascript
const PORT = 3000; // Change this
```

### CORS
Backend allows all origins for development. For production, update CORS configuration in `server/index.js`.

## 🐛 Troubleshooting

### "Port already in use"
- Kill the process using the port
- Or change the port in configuration

### "Cannot connect to API"
- Ensure backend is running on port 3000
- Check CORS configuration
- Verify API_URL in `LiveDemo.jsx`

### "Image upload fails"
- Check file size (max 5MB)
- Verify file format (PNG/JPG/JPEG)
- Ensure backend is running
- Check server/uploads/ directory permissions

### "Charts not displaying"
- Clear browser cache
- Check console for errors
- Verify Chart.js is installed

### "Dark mode not working"
- Clear localStorage
- Check browser console
- Refresh the page

## 📦 Building for Production

```bash
# Build frontend
npm run build

# Output will be in dist/ directory
# Serve with any static file server
```

## 🔐 Privacy & Security

- **No permanent storage** - Images deleted after 30 minutes
- **No database** - In-memory storage only
- **CORS** - Configured for localhost development
- **File validation** - Size and type checking
- **Secure headers** - Can be added for production

## 📝 Development Notes

### Adding New Features
1. Create component in `src/components/`
2. Create corresponding CSS file
3. Import in `App.jsx`
4. Add to navigation if needed

### Styling Guidelines
- Use CSS variables defined in `index.css`
- Follow mobile-first approach
- Maintain 8px spacing system
- Use transitions for all interactive elements

### API Development
- Add new endpoints in `server/index.js`
- Use proper HTTP status codes
- Include error handling
- Document in README

## 🎓 Learning Resources

- **React:** https://react.dev
- **Vite:** https://vitejs.dev
- **Framer Motion:** https://www.framer.com/motion
- **Chart.js:** https://www.chartjs.org

## 📄 License

Educational and research purposes only. Not for clinical use without proper validation and regulatory approval.

## 🙏 Acknowledgments

- APTOS 2019 Dataset
- PyTorch & Flower (Flwr)
- React & Vite communities
- All open-source contributors

## 📞 Support

For issues with:
- **Web app:** Check browser console and network tab
- **Backend:** Check server logs in terminal
- **ML model:** See main federated-dr/ README

---

Built with ❤️ for advancing privacy-preserving AI in healthcare

**Enjoy exploring FedLearnDR!** 🚀🏥
