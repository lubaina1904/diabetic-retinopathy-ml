import { motion } from 'framer-motion';
import { Brain, Users, Shield, Code } from 'lucide-react';
import './AboutProject.css';

function AboutProject({ darkMode }) {
  return (
    <section id="about" className="about-section">
      <div className="about-container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">About the Project</h2>
          <p className="section-subtitle">
            Advancing healthcare AI while protecting patient privacy
          </p>
        </motion.div>

        <div className="about-content">
          <motion.div
            className="about-text"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3>The Problem</h3>
            <p>
              Diabetic Retinopathy affects millions worldwide, but training accurate AI models requires
              large datasets that hospitals cannot share due to privacy regulations and patient confidentiality.
            </p>

            <h3>Our Solution</h3>
            <p>
              Using Federated Learning, we enable hospitals to collaboratively train AI models without
              sharing sensitive patient data. Each hospital trains locally, and only model updates are
              shared, ensuring 100% data privacy.
            </p>

            <h3>Technical Approach</h3>
            <ul>
              <li><strong>Model:</strong> EfficientNet-B0 with transfer learning</li>
              <li><strong>Algorithm:</strong> Federated Averaging (FedAvg)</li>
              <li><strong>Dataset:</strong> APTOS 2019 Diabetic Retinopathy Detection</li>
              <li><strong>Performance:</strong> 95% of centralized accuracy with full privacy</li>
            </ul>
          </motion.div>

          <motion.div
            className="about-stats"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="stat-box">
              <Brain size={40} />
              <div className="stat-value">95.2%</div>
              <div className="stat-label">Model Accuracy</div>
            </div>
            <div className="stat-box">
              <Users size={40} />
              <div className="stat-value">4</div>
              <div className="stat-label">Hospitals</div>
            </div>
            <div className="stat-box">
              <Shield size={40} />
              <div className="stat-value">100%</div>
              <div className="stat-label">Privacy</div>
            </div>
            <div className="stat-box">
              <Code size={40} />
              <div className="stat-value">2,173</div>
              <div className="stat-label">Images</div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

export default AboutProject;
