const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png/;
    const ext = allowed.test(path.extname(file.originalname).toLowerCase());
    const mime = allowed.test(file.mimetype);
    if (ext && mime) {
      cb(null, true);
    } else {
      cb(new Error('Only JPEG and PNG images allowed'));
    }
  }
});

if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

const imageCache = new Map();

const hospitals = [
  { id: 1, name: 'Metropolitan General Hospital', type: 'General', imageCount: 645 },
  { id: 2, name: 'Vision Care Specialized Center', type: 'Specialized', imageCount: 523 },
  { id: 3, name: 'Urban Health Medical Center', type: 'Urban', imageCount: 512 },
  { id: 4, name: 'Advanced Research Institute', type: 'Research', imageCount: 493 }
];

const severityLabels = [
  'No DR (Healthy)',
  'Mild DR',
  'Moderate DR',
  'Severe DR',
  'Proliferative DR'
];

const mockMetrics = {
  totalImages: 2173,
  hospitalCount: 4,
  accuracy: 95.2,
  privacyScore: 100,
  comparisonData: {
    centralized: 96.8,
    federated: 95.2,
    difference: 1.6
  },
  trainingCurves: {
    rounds: Array.from({length: 30}, (_, i) => i + 1),
    fedAvgAccuracy: [45, 52, 58, 64, 69, 73, 76, 79, 81, 83, 85, 86, 87, 88, 89, 90, 91, 91.5, 92, 92.5, 93, 93.3, 93.6, 94, 94.3, 94.6, 94.9, 95, 95.1, 95.2],
    centralizedAccuracy: Array(30).fill(96.8),
    fedAvgLoss: [1.8, 1.6, 1.45, 1.32, 1.21, 1.12, 1.04, 0.97, 0.91, 0.86, 0.81, 0.77, 0.73, 0.69, 0.66, 0.63, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.49, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41]
  },
  perHospitalPerformance: {
    hospitals: hospitals.map(h => h.name),
    accuracy: [94.2, 96.1, 95.8, 94.6],
    kappa: [0.89, 0.93, 0.91, 0.90]
  },
  confusionMatrix: [
    [452, 12, 3, 0, 0],
    [18, 321, 24, 2, 0],
    [5, 28, 298, 15, 3],
    [0, 3, 19, 187, 8],
    [0, 0, 2, 12, 161]
  ],
  metrics: {
    accuracy: 95.2,
    kappaScore: 0.91,
    trainingTime: 142,
    privacyLevel: 100
  }
};

app.post('/api/upload', upload.single('image'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const imageId = Date.now().toString();
    const imageUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;

    imageCache.set(imageId, {
      filename: req.file.filename,
      path: req.file.path,
      uploadedAt: Date.now()
    });

    setTimeout(() => {
      const cached = imageCache.get(imageId);
      if (cached && fs.existsSync(cached.path)) {
        fs.unlinkSync(cached.path);
      }
      imageCache.delete(imageId);
    }, 30 * 60 * 1000);

    res.json({
      imageId,
      preview_url: imageUrl,
      message: 'Image uploaded successfully'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/analyze', (req, res) => {
  try {
    const { imageId, hospitalId } = req.body;

    if (!imageId || !hospitalId) {
      return res.status(400).json({ error: 'Missing imageId or hospitalId' });
    }

    if (!imageCache.has(imageId)) {
      return res.status(404).json({ error: 'Image not found or expired' });
    }

    setTimeout(() => {
      const severity = Math.floor(Math.random() * 5);
      const confidence = 75 + Math.random() * 20;

      let riskLevel;
      if (severity === 0) riskLevel = 'low';
      else if (severity === 1) riskLevel = 'low';
      else if (severity === 2) riskLevel = 'medium';
      else if (severity === 3) riskLevel = 'high';
      else riskLevel = 'critical';

      const result = {
        analysisId: `ANL-${Date.now()}`,
        severity,
        severityLabel: severityLabels[severity],
        confidence: parseFloat(confidence.toFixed(2)),
        riskLevel,
        hospitalId,
        hospitalName: hospitals.find(h => h.id === parseInt(hospitalId))?.name || 'Unknown',
        timestamp: new Date().toISOString(),
        details: {
          microaneurysms: severity > 0,
          hemorrhages: severity > 1,
          hardExudates: severity > 2,
          softExudates: severity > 2,
          neovascularization: severity > 3
        }
      };

      res.json(result);
    }, 1500);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/hospitals', (req, res) => {
  res.json(hospitals);
});

app.get('/api/metrics', (req, res) => {
  res.json(mockMetrics);
});

app.get('/api/results/:id', (req, res) => {
  res.json({
    message: 'Result retrieved',
    analysisId: req.params.id
  });
});

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message || 'Something went wrong!' });
});

app.listen(PORT, () => {
  console.log(`🚀 Federated Learning API running on http://localhost:${PORT}`);
  console.log(`📊 Ready to analyze retinal images with privacy preservation`);
});
