import React, { useRef, useState } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { ArrowUpTrayIcon, DocumentTextIcon, XMarkIcon } from '@heroicons/react/24/outline';

const FileUploader = () => {
    const fileInputRef = useRef(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [results, setResults] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const handleFileSelect = (file) => {
        // Check if file is CSV or Excel
        const validExtensions = ['.csv', '.xlsx', '.xls'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(fileExtension)) {
        toast.error('Please upload a CSV or Excel file');
        return;
        }
        
        setSelectedFile(file);
    };

    const openFilePicker = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
        toast.error('Please select a file first');
        return;
        }

        setIsUploading(true);
        setResults(null);
        
        try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await axios.post('/api/upload-analyze-file', formData, {
            headers: {
            'Content-Type': 'multipart/form-data'
            }
        });
        
        setResults(response.data);
        toast.success('File analyzed successfully');
        } catch (error) {
        console.error('Upload error:', error);
        toast.error(error.response?.data?.detail || 'Error analyzing file');
        } finally {
        setIsUploading(false);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
        setResults(null);
    };

    return (
        <div className="rounded-lg bg-white shadow p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">File Analysis</h2>
        
        <div
            className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
            dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            <div className="flex flex-col items-center justify-center space-y-3">
            <ArrowUpTrayIcon className="h-12 w-12 text-gray-400" />
            <p className="text-gray-700">Drag and drop your CSV or Excel file here</p>
            <p className="text-sm text-gray-500">or</p>
            <button
                onClick={openFilePicker}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                type="button"
            >
                Browse Files
            </button>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".csv,.xlsx,.xls"
                className="hidden"
            />
            <p className="text-xs text-gray-500 mt-2">Supports CSV and Excel files</p>
            </div>
        </div>

        {selectedFile && (
            <div className="mt-4 p-3 bg-gray-50 rounded-md flex items-center justify-between">
            <div className="flex items-center">
                <DocumentTextIcon className="h-5 w-5 text-blue-500 mr-2" />
                <span className="text-sm font-medium">{selectedFile.name}</span>
                <span className="ml-2 text-xs text-gray-500">
                {(selectedFile.size / 1024).toFixed(2)} KB
                </span>
            </div>
            <button
                onClick={clearFile}
                className="text-gray-500 hover:text-gray-700"
                aria-label="Remove file"
            >
                <XMarkIcon className="h-5 w-5" />
            </button>
            </div>
        )}

        <div className="mt-4">
            <button
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className={`w-full py-2 px-4 rounded-md text-white font-medium ${
                !selectedFile || isUploading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-green-600 hover:bg-green-700"
            }`}
            >
            {isUploading ? "Analyzing..." : "Analyze File"}
            </button>
        </div>

        {results && (
            <div className="mt-6">
            <div className="mb-4 p-4 bg-gray-50 rounded-md">
                <h3 className="font-medium text-gray-900 mb-2">Analysis Summary</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                <div>File Type: <span className="font-medium">{results.file_type.toUpperCase()}</span></div>
                <div>Total Rows: <span className="font-medium">{results.total_rows}</span></div>
                <div>Successful: <span className="font-medium text-green-600">{results.successful_rows}</span></div>
                <div>Failed: <span className="font-medium text-red-600">{results.failed_rows}</span></div>
                <div>Processing Time: <span className="font-medium">{(results.processing_time_ms / 1000).toFixed(2)}s</span></div>
                </div>
            </div>

            <h3 className="font-medium text-gray-900 mb-2">Results</h3>
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                    <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Row</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Text</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {results.results.map((row) => (
                    <tr key={row.row_number} className={row.error ? "bg-red-50" : ""}>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{row.row_number}</td>
                        <td className="px-3 py-2 text-sm text-gray-500 max-w-xs truncate">
                        {row.text || "N/A"}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                        {row.error ? (
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                            Failed
                            </span>
                        ) : (
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            Success
                            </span>
                        )}
                        </td>
                        <td className="px-3 py-2 text-sm text-gray-500">
                        {row.error ? (
                            <span className="text-red-600">{row.error}</span>
                        ) : (
                            <details className="cursor-pointer">
                            <summary className="text-blue-600 hover:text-blue-800">View analysis</summary>
                            <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-x-auto">
                                {JSON.stringify(row.analysis_result, null, 2)}
                            </pre>
                            </details>
                        )}
                        </td>
                    </tr>
                    ))}
                </tbody>
                </table>
            </div>
            </div>
        )}
        </div>
    );
};

export default FileUploader; 