import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Eye, AlertCircle, CheckCircle, Loader, Download } from 'lucide-react';
import axios from 'axios';
import './LiveDemo.css';

const API_URL = 'http://localhost:3000/api';

function LiveDemo({ darkMode }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [hospital, setHospital] = useState('');
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageId, setImageId] = useState(null);

  useState(() => {
    axios.get(`${API_URL}/hospitals`)
      .then(res => setHospitals(res.data))
      .catch(err => console.error('Failed to load hospitals', err));
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.size <= 5 * 1024 * 1024) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('File size must be less than 5MB');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !hospital) {
      setError('Please select an image and hospital');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const uploadRes = await axios.post(`${API_URL}/upload`, formData);
      const { imageId } = uploadRes.data;
      setImageId(imageId);

      const analysisRes = await axios.post(`${API_URL}/analyze`, {
        imageId,
        hospitalId: parseInt(hospital)
      });

      setResult(analysisRes.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#ef4444',
      critical: '#dc2626'
    };
    return colors[risk] || '#64748b';
  };

  return (
    <section id="demo" className="demo-section">
      <div className="demo-container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Live Demo</h2>
          <p className="section-subtitle">
            Upload a retinal image and see our federated learning model in action
          </p>
        </motion.div>

        <div className="demo-grid">
          <motion.div
            className="upload-panel"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div
              className={`upload-area ${preview ? 'has-image' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <button
                    className="change-image-btn"
                    onClick={() => {
                      setPreview(null);
                      setSelectedFile(null);
                      setResult(null);
                    }}
                  >
                    Change Image
                  </button>
                </div>
              ) : (
                <label className="upload-label">
                  <input
                    type="file"
                    accept="image/jpeg,image/jpg,image/png"
                    onChange={handleFileSelect}
                    className="file-input"
                  />
                  <Upload size={48} />
                  <p className="upload-text">
                    Drag & drop or <span className="upload-link">browse</span>
                  </p>
                  <p className="upload-hint">Max 5MB | PNG, JPG, JPEG</p>
                </label>
              )}
            </div>

            <div className="hospital-select">
              <label className="input-label">Select Hospital</label>
              <select
                value={hospital}
                onChange={(e) => setHospital(e.target.value)}
                className="select-input"
              >
                <option value="">Choose a hospital...</option>
                {hospitals.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name} ({h.type})
                  </option>
                ))}
              </select>
            </div>

            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={loading || !selectedFile || !hospital}
            >
              {loading ? (
                <>
                  <Loader className="spinning" size={20} />
                  Analyzing...
                </>
              ) : (
                <>
                  <Eye size={20} />
                  Analyze Image
                </>
              )}
            </button>

            {error && (
              <motion.div
                className="error-message"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <AlertCircle size={20} />
                {error}
              </motion.div>
            )}

            <div className="privacy-notice">
              <AlertCircle size={16} />
              <span>
                Your image is processed locally and deleted after 30 minutes. No data is permanently stored.
              </span>
            </div>
          </motion.div>

          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                className="result-panel"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <div className="result-header">
                  <CheckCircle size={24} color="#10b981" />
                  <h3>Analysis Complete</h3>
                </div>

                <div className="result-main">
                  <div className="severity-display">
                    <div className="severity-label">Diagnosis</div>
                    <div className="severity-value">{result.severityLabel}</div>
                    <div className="severity-number">Grade {result.severity}/4</div>
                  </div>

                  <div className="confidence-meter">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-bar">
                      <motion.div
                        className="confidence-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                      />
                    </div>
                    <div className="confidence-value">{result.confidence.toFixed(1)}%</div>
                  </div>

                  <div className="risk-badge" style={{ borderColor: getRiskColor(result.riskLevel) }}>
                    <span
                      className="risk-indicator"
                      style={{ background: getRiskColor(result.riskLevel) }}
                    />
                    <span className="risk-text">
                      Risk Level: <strong>{result.riskLevel.toUpperCase()}</strong>
                    </span>
                  </div>
                </div>

                <div className="result-details">
                  <h4>Clinical Findings</h4>
                  <div className="findings-grid">
                    {Object.entries(result.details).map(([key, value]) => (
                      <div key={key} className="finding-item">
                        <span className={`finding-indicator ${value ? 'positive' : 'negative'}`} />
                        <span className="finding-name">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="result-meta">
                  <div className="meta-item">
                    <span className="meta-label">Hospital:</span>
                    <span className="meta-value">{result.hospitalName}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Analysis ID:</span>
                    <span className="meta-value">{result.analysisId}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Timestamp:</span>
                    <span className="meta-value">
                      {new Date(result.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

export default LiveDemo;
