import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import './TechnicalDetails.css';

function TechnicalDetails({ darkMode }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <section id="technical" className="technical-section">
      <div className="technical-container">
        <motion.button
          className="technical-toggle"
          onClick={() => setIsOpen(!isOpen)}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <span>Technical Details</span>
          <motion.div animate={{ rotate: isOpen ? 180 : 0 }}>
            <ChevronDown />
          </motion.div>
        </motion.button>

        <AnimatePresence>
          {isOpen && (
            <motion.div
              className="technical-content"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="tech-grid">
                <div className="tech-card">
                  <h3>Model Architecture</h3>
                  <p><strong>Base Model:</strong> EfficientNet-B0</p>
                  <p><strong>Pre-training:</strong> ImageNet weights</p>
                  <p><strong>Output:</strong> 5-class classification (DR severity 0-4)</p>
                  <p><strong>Parameters:</strong> ~5.3M trainable</p>
                </div>

                <div className="tech-card">
                  <h3>Federated Learning</h3>
                  <p><strong>Algorithm:</strong> FedAvg (Federated Averaging)</p>
                  <p><strong>Rounds:</strong> 30 communication rounds</p>
                  <p><strong>Local Epochs:</strong> 3 per round</p>
                  <p><strong>Aggregation:</strong> Weighted by dataset size</p>
                </div>

                <div className="tech-card">
                  <h3>Dataset</h3>
                  <p><strong>Source:</strong> APTOS 2019 Blindness Detection</p>
                  <p><strong>Total Images:</strong> 2,173 retinal fundus images</p>
                  <p><strong>Distribution:</strong> Non-IID across 4 hospitals</p>
                  <p><strong>Classes:</strong> 0 (No DR) to 4 (Proliferative DR)</p>
                </div>

                <div className="tech-card">
                  <h3>Implementation</h3>
                  <p><strong>Framework:</strong> PyTorch, Flower (Flwr)</p>
                  <p><strong>Privacy:</strong> Opacus for differential privacy</p>
                  <p><strong>Frontend:</strong> React + Vite</p>
                  <p><strong>Backend:</strong> Express.js + Node</p>
                </div>
              </div>

              <div className="code-snippet">
                <h4>Sample Code</h4>
                <pre><code>{`# Federated Learning Client
class HospitalClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader)
        return get_parameters(self.model), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate(self.model, self.val_loader)
        return loss, len(self.val_loader), {"accuracy": accuracy}`}</code></pre>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}

export default TechnicalDetails;
