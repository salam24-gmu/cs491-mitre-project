import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import FileUploader from './fileuploader';

function App() {
  return (
    <div>
      <h1 className="text-center text-2xl font-bold mt-4">Mitre Data Analyzer Demo</h1>
      <FileUploader />
    </div>
  );
}

export default App
