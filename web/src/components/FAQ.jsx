import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import './FAQ.css';

function FAQ({ darkMode }) {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    {
      question: 'What is Federated Learning?',
      answer: 'Federated Learning is a machine learning approach where multiple parties collaborate to train a shared model without exchanging their raw data. Each party trains the model locally, and only model updates are shared.'
    },
    {
      question: 'How is patient privacy protected?',
      answer: 'Patient data never leaves the hospital. Only encrypted model parameters are shared during training. Additionally, differential privacy techniques can add noise to further protect individual privacy.'
    },
    {
      question: 'Is federated learning as accurate as traditional methods?',
      answer: 'Our results show that federated learning achieves 95% of centralized accuracy while maintaining 100% data privacy. The small accuracy trade-off is often worthwhile for the significant privacy benefits.'
    },
    {
      question: 'What is Diabetic Retinopathy?',
      answer: 'Diabetic Retinopathy is a diabetes complication that affects the eyes. It is caused by damage to blood vessels in the retina and can lead to blindness if untreated. Early detection through AI can save vision.'
    },
    {
      question: 'Can I use this model in production?',
      answer: 'This is a research prototype demonstrating federated learning concepts. For production medical use, the model would need regulatory approval, extensive validation, and integration with clinical workflows.'
    }
  ];

  return (
    <section id="faq" className="faq-section">
      <div className="faq-container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Frequently Asked Questions</h2>
          <p className="section-subtitle">
            Everything you need to know about our federated learning approach
          </p>
        </motion.div>

        <div className="faq-list">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              className="faq-item"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <button
                className="faq-question"
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
              >
                <span>{faq.question}</span>
                <motion.div animate={{ rotate: openIndex === index ? 180 : 0 }}>
                  <ChevronDown />
                </motion.div>
              </button>
              <AnimatePresence>
                {openIndex === index && (
                  <motion.div
                    className="faq-answer"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <p>{faq.answer}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default FAQ;
