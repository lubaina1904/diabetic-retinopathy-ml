import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HeroSection from './components/HeroSection';
import HowItWorks from './components/HowItWorks';
import LiveDemo from './components/LiveDemo';
import ResultsDashboard from './components/ResultsDashboard';
import AboutProject from './components/AboutProject';
import TechnicalDetails from './components/TechnicalDetails';
import FAQ from './components/FAQ';
import Footer from './components/Footer';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setMobileMenuOpen(false);
    }
  };

  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>
      <Navigation
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        mobileMenuOpen={mobileMenuOpen}
        setMobileMenuOpen={setMobileMenuOpen}
        scrollToSection={scrollToSection}
      />

      <main className="main-content">
        <HeroSection scrollToSection={scrollToSection} darkMode={darkMode} />
        <HowItWorks darkMode={darkMode} />
        <LiveDemo darkMode={darkMode} />
        <ResultsDashboard darkMode={darkMode} />
        <AboutProject darkMode={darkMode} />
        <TechnicalDetails darkMode={darkMode} />
        <FAQ darkMode={darkMode} />
      </main>

      <Footer darkMode={darkMode} scrollToSection={scrollToSection} />
    </div>
  );
}

export default App;
