# Getting Started with FedLearnDR Web App

## Step 1: Install Dependencies

```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd server
npm install
cd ..
```

## Step 2: Start the Application

### Option A: Two Separate Terminals

**Terminal 1 - Backend:**
```bash
cd server
npm start
```
Backend will run on: http://localhost:3000

**Terminal 2 - Frontend:**
```bash
npm run dev
```
Frontend will run on: http://localhost:5173

### Option B: Single Script (Unix/Linux/Mac)
```bash
chmod +x start.sh
./start.sh
```

## Step 3: Use the Application

1. Open your browser to **http://localhost:5173**
2. Scroll to the "Live Demo" section
3. Upload a retinal image (PNG, JPG, JPEG - max 5MB)
4. Select a hospital from the dropdown
5. Click "Analyze Image"
6. View the results with DR severity and confidence

## Features to Explore

- ✅ Dark mode toggle (top right)
- ✅ Interactive demo with image upload
- ✅ Results dashboard with charts
- ✅ How It Works explanation
- ✅ FAQ section
- ✅ Technical details (collapsible)

## Troubleshooting

**Port already in use?**
- Frontend (5173): Change in `vite.config.js`
- Backend (3000): Change PORT in `server/index.js`

**CORS errors?**
- Ensure backend is running on port 3000
- Check browser console for errors

**Image upload fails?**
- Check file size (max 5MB)
- Ensure format is PNG, JPG, or JPEG
- Backend must be running

## Tech Stack

- **Frontend:** React 18, Vite, Framer Motion, Chart.js
- **Backend:** Express.js, Multer, CORS
- **Styling:** Custom CSS with CSS Variables
- **Icons:** Lucide React

Enjoy exploring privacy-preserving AI for healthcare! 🏥🔐
