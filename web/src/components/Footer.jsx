import { Activity, Github, FileText, BookOpen, Mail } from 'lucide-react';
import './Footer.css';

function Footer({ darkMode, scrollToSection }) {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-grid">
          <div className="footer-brand">
            <div className="footer-logo">
              <Activity className="logo-icon" />
              <span className="logo-text">FedLearn<span className="logo-accent">DR</span></span>
            </div>
            <p className="footer-tagline">
              Privacy-preserving AI for healthcare. Collaborative learning without compromising patient data.
            </p>
            <div className="footer-social">
              <a href="https://github.com/lubaina1904/diabetic-retinopathy-ml" target="_blank" rel="noopener noreferrer">
                <Github />
              </a>
              <a href="#" target="_blank" rel="noopener noreferrer">
                <FileText />
              </a>
              <a href="#" target="_blank" rel="noopener noreferrer">
                <BookOpen />
              </a>
            </div>
          </div>

          <div className="footer-links">
            <h4>Navigation</h4>
            <button onClick={() => scrollToSection('hero')}>Home</button>
            <button onClick={() => scrollToSection('how-it-works')}>How It Works</button>
            <button onClick={() => scrollToSection('demo')}>Demo</button>
            <button onClick={() => scrollToSection('results')}>Results</button>
          </div>

          <div className="footer-links">
            <h4>Resources</h4>
            <a href="https://github.com/lubaina1904/diabetic-retinopathy-ml" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
            <a href="#" target="_blank" rel="noopener noreferrer">Documentation</a>
            <a href="#" target="_blank" rel="noopener noreferrer">Research Paper</a>
            <button onClick={() => scrollToSection('faq')}>FAQ</button>
          </div>

          <div className="footer-links">
            <h4>Tech Stack</h4>
            <p>PyTorch & Flower</p>
            <p>React & Vite</p>
            <p>Express.js</p>
            <p>Chart.js</p>
          </div>
        </div>

        <div className="footer-bottom">
          <p>&copy; {currentYear} FedLearnDR. Built for educational and research purposes.</p>
          <p className="footer-disclaimer">
            This is a research prototype. Not for clinical use without proper validation and regulatory approval.
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
