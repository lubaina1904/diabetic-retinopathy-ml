import { motion } from 'framer-motion';
import { Database, Users, Shield, TrendingUp } from 'lucide-react';
import './HowItWorks.css';

function HowItWorks({ darkMode }) {
  const steps = [
    {
      icon: Database,
      title: 'Data Stays Local',
      description: 'Each hospital keeps their patient data secure on their own servers. No raw data ever leaves the hospital.'
    },
    {
      icon: Users,
      title: 'Local Training',
      description: 'Each hospital trains an AI model on their local data independently, learning patterns specific to their patients.'
    },
    {
      icon: Shield,
      title: 'Secure Aggregation',
      description: 'Only model updates (not data) are shared with a central server and combined using advanced encryption.'
    },
    {
      icon: TrendingUp,
      title: 'Global Model',
      description: 'The combined knowledge creates a powerful global model that benefits from all hospitals without compromising privacy.'
    }
  ];

  return (
    <section id="how-it-works" className="how-it-works-section">
      <div className="how-container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">How Federated Learning Works</h2>
          <p className="section-subtitle">
            Collaborative AI without compromising patient privacy
          </p>
        </motion.div>

        <div className="steps-grid">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              className="step-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2 }}
            >
              <div className="step-number">{index + 1}</div>
              <div className="step-icon">
                <step.icon />
              </div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-description">{step.description}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          className="comparison-toggle"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.8 }}
        >
          <div className="comparison-card traditional">
            <h4>Traditional ML</h4>
            <p>Centralized data storage</p>
            <p className="risk-text">Privacy risks</p>
          </div>
          <div className="vs-badge">VS</div>
          <div className="comparison-card federated">
            <h4>Federated Learning</h4>
            <p>Distributed training</p>
            <p className="safe-text">100% Privacy preserved</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default HowItWorks;
