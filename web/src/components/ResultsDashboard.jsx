import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale, Title, Tooltip, Legend } from 'chart.js';
import { Download, TrendingUp, Award, Clock, Shield } from 'lucide-react';
import axios from 'axios';
import './ResultsDashboard.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale, Title, Tooltip, Legend);

const API_URL = 'http://localhost:3000/api';

function ResultsDashboard({ darkMode }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/metrics`)
      .then(res => setMetrics(res.data))
      .catch(err => console.error('Failed to load metrics', err));
  }, []);

  if (!metrics) {
    return (
      <section id="results" className="results-section">
        <div className="results-container">
          <div className="loading-skeleton">Loading results...</div>
        </div>
      </section>
    );
  }

  const trainingData = {
    labels: metrics.trainingCurves.rounds,
    datasets: [
      {
        label: 'Federated Learning',
        data: metrics.trainingCurves.fedAvgAccuracy,
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.1)',
        tension: 0.4
      },
      {
        label: 'Centralized (Baseline)',
        data: metrics.trainingCurves.centralizedAccuracy,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderDash: [5, 5],
        tension: 0.4
      }
    ]
  };

  const comparisonData = {
    labels: ['Accuracy (%)'],
    datasets: [
      {
        label: 'Centralized',
        data: [metrics.comparisonData.centralized],
        backgroundColor: 'rgba(16, 185, 129, 0.8)'
      },
      {
        label: 'Federated',
        data: [metrics.comparisonData.federated],
        backgroundColor: 'rgba(37, 99, 235, 0.8)'
      }
    ]
  };

  const hospitalData = {
    labels: metrics.perHospitalPerformance.hospitals.map(h => h.split(' ')[0]),
    datasets: [
      {
        label: 'Accuracy',
        data: metrics.perHospitalPerformance.accuracy,
        backgroundColor: 'rgba(37, 99, 235, 0.6)',
        borderColor: '#2563eb',
        borderWidth: 2
      }
    ]
  };

  return (
    <section id="results" className="results-section">
      <div className="results-container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Results Dashboard</h2>
          <p className="section-subtitle">
            Comprehensive analysis comparing centralized vs federated learning
          </p>
        </motion.div>

        <div className="metrics-cards">
          {[
            { icon: TrendingUp, label: 'Accuracy', value: `${metrics.metrics.accuracy}%`, color: '#2563eb' },
            { icon: Award, label: 'Kappa Score', value: metrics.metrics.kappaScore.toFixed(2), color: '#10b981' },
            { icon: Clock, label: 'Training Time', value: `${metrics.metrics.trainingTime}min`, color: '#f59e0b' },
            { icon: Shield, label: 'Privacy Level', value: `${metrics.metrics.privacyLevel}%`, color: '#059669' }
          ].map((metric, idx) => (
            <motion.div
              key={idx}
              className="metric-dashboard-card"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
            >
              <div className="metric-icon-wrap" style={{ background: `${metric.color}20` }}>
                <metric.icon style={{ color: metric.color }} />
              </div>
              <div className="metric-info">
                <div className="metric-value-large">{metric.value}</div>
                <div className="metric-label-small">{metric.label}</div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="charts-grid">
          <motion.div
            className="chart-card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h3 className="chart-title">Training Progress</h3>
            <Line data={trainingData} options={{ responsive: true, maintainAspectRatio: true }} />
          </motion.div>

          <motion.div
            className="chart-card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="chart-title">Accuracy Comparison</h3>
            <Bar data={comparisonData} options={{ responsive: true, maintainAspectRatio: true }} />
          </motion.div>

          <motion.div
            className="chart-card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
          >
            <h3 className="chart-title">Per-Hospital Performance</h3>
            <Radar data={hospitalData} options={{ responsive: true, maintainAspectRatio: true }} />
          </motion.div>
        </div>
      </div>
    </section>
  );
}

export default ResultsDashboard;
