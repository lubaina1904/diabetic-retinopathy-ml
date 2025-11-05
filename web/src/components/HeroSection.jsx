import { motion } from 'framer-motion';
import { ArrowRight, Users, Database, TrendingUp, Shield } from 'lucide-react';
import './HeroSection.css';

function HeroSection({ scrollToSection, darkMode }) {
  const metrics = [
    { icon: Users, value: '4', label: 'Hospitals Collaborating' },
    { icon: Database, value: '2,173', label: 'Images Analyzed' },
    { icon: TrendingUp, value: '95%', label: 'of Centralized Performance' },
    { icon: Shield, value: '100%', label: 'Privacy Preserved' }
  ];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <section id="hero" className="hero-section">
      <div className="hero-background">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <div className="hero-container">
        <motion.div
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div
            className="hero-badge"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Shield size={16} />
            <span>Privacy-First AI Healthcare</span>
          </motion.div>

          <h1 className="hero-title">
            Privacy-Preserving AI for
            <span className="hero-title-gradient"> Healthcare</span>
          </h1>

          <p className="hero-subtitle">
            Collaborative Diabetic Retinopathy Detection Across Hospitals Without Sharing Patient Data
          </p>

          <div className="hero-buttons">
            <button
              className="btn btn-primary"
              onClick={() => scrollToSection('demo')}
            >
              Try Demo
              <ArrowRight size={20} />
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => scrollToSection('about')}
            >
              View Research
            </button>
          </div>
        </motion.div>

        <motion.div
          className="metrics-grid"
          variants={container}
          initial="hidden"
          animate="show"
        >
          {metrics.map((metric, index) => (
            <motion.div
              key={index}
              className="metric-card"
              variants={item}
              whileHover={{ scale: 1.05, y: -5 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <div className="metric-icon">
                <metric.icon />
              </div>
              <div className="metric-content">
                <div className="metric-value">{metric.value}</div>
                <div className="metric-label">{metric.label}</div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      <motion.div
        className="scroll-indicator"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
      >
        <div className="scroll-mouse"></div>
      </motion.div>
    </section>
  );
}

export default HeroSection;
