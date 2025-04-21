import React from 'react';
import Layout from './components/Layout';
import ThreatDetectionForm from './components/ThreatDetectionForm';
import FileUploader from './components/FileUploader';

function App() {
  return (
    <Layout>
      <div className="space-y-4 max-w-7xl mx-auto w-full">
        <section id="threat-detection">
          <ThreatDetectionForm />
        </section>
        
        <section id="file-analysis">
          <FileUploader />
        </section>
      </div>
    </Layout>
  );
}

export default App;
