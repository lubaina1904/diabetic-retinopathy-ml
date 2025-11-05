# FedLearnDR - Privacy-Preserving AI for Healthcare

A stunning web application demonstrating **Federated Learning** for **Diabetic Retinopathy Detection** across multiple hospitals without sharing patient data.

## 🚀 Quick Start

### Installation

1. Install frontend dependencies:
```bash
cd web
npm install
```

2. Install backend dependencies:
```bash
cd server
npm install
```

### Running the Application

**Terminal 1 (Backend):**
```bash
cd web/server
npm start
```

**Terminal 2 (Frontend):**
```bash
cd web
npm run dev
```

### Access

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:3000

## ✨ Features

- 🎨 Beautiful modern UI with dark mode
- 📱 Fully responsive (mobile, tablet, desktop)
- 🎬 Smooth animations with Framer Motion
- 📊 Interactive charts and visualizations
- 🔐 Privacy-preserving federated learning demo
- 🖼️ Drag & drop image upload
- 📈 Real-time analysis results
- 🏥 Multi-hospital simulation

## 🛠️ Technology Stack

### Frontend
- React 18 + Vite
- Framer Motion
- Chart.js
- Axios
- Lucide React Icons

### Backend
- Node.js + Express
- Multer for file uploads
- In-memory storage
- CORS enabled

## 📋 API Endpoints

- `POST /api/upload` - Upload retinal image
- `POST /api/analyze` - Analyze image
- `GET /api/hospitals` - Get hospital list
- `GET /api/metrics` - Get dashboard metrics

## 🎨 Design

- Medical aesthetic: Blues (#2563eb) & Greens (#10b981)
- Inter font family
- WCAG 2.1 AA compliant
- Dark mode support

Built with ❤️ for advancing privacy-preserving AI in healthcare
