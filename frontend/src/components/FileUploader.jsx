import React, { useRef, useState } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { ArrowUpTrayIcon, DocumentTextIcon, XMarkIcon, ExclamationTriangleIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

const FileUploader = () => {
    const fileInputRef = useRef(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [results, setResults] = useState(null);
    const [plotPaths, setPlotPaths] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [resultsExpanded, setResultsExpanded] = useState(false);
    const [fullScreenImage, setFullScreenImage] = useState(null);

    const handleFileSelect = (file) => {
        // Check if file is CSV or Excel
        const validExtensions = ['.csv', '.xlsx', '.xls'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(fileExtension)) {
        toast.error('Please upload a CSV or Excel file');
        return;
        }
        
        setSelectedFile(file);
        setResults(null);
        setPlotPaths(null);
        setResultsExpanded(false);
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
        setPlotPaths(null);
        setResultsExpanded(false);
        
        try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await axios.post('/api/upload-analyze-file', formData, {
            headers: {
            'Content-Type': 'multipart/form-data'
            }
        });
        
        setResults(response.data);
        if (response.data.plot_paths) {
            setPlotPaths(response.data.plot_paths);
        }
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
        setPlotPaths(null);
        setResultsExpanded(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const toggleResultsExpanded = () => {
        setResultsExpanded(prev => !prev);
    };

    // Calculate classification summary counts
    const getClassificationSummary = () => {
        if (!results || !results.results) return null;
        
        const maliciousCount = results.results.filter(row => row.predicted_class === 'malicious').length;
        const nonMaliciousCount = results.results.filter(row => row.predicted_class === 'non-malicious').length;
        const unclassifiedCount = results.results.filter(row => row.predicted_class === null && !row.error).length;
        
        return { maliciousCount, nonMaliciousCount, unclassifiedCount };
    };

    // Function to render the plots
    const renderPlots = () => {
        if (!plotPaths) return null;

        if (plotPaths.error) {
            return (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
                    <div className="flex items-center">
                        <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2 flex-shrink-0" />
                        <h3 className="text-sm font-medium text-red-800">Plot Generation Failed</h3>
                    </div>
                    <p className="text-sm text-red-700 mt-1">Error: {plotPaths.error}</p>
                </div>
            );
        }

        const requestId = plotPaths.request_id;
        if (!requestId) {
            return (
                <div className="mt-6 text-sm text-gray-500">
                    No plots generated for this analysis.
                </div>
            );
        }

        // Helper to create image element
        const createImage = (filename, altText) => {
            if (!filename) return null;
            // Use the backend's dedicated API endpoint to get the image
            const url = `/api/plots/${requestId}/${filename}`;
            return (
                <div className="border rounded-md p-2 bg-white shadow-sm">
                    <img 
                        src={url} 
                        alt={altText} 
                        className="max-w-full h-auto mx-auto cursor-pointer transition hover:opacity-90" 
                        loading="lazy"
                        onClick={() => setFullScreenImage({ url, altText })}
                        onError={(e) => { 
                            console.error(`Failed to load image: ${url}`);
                            e.target.style.display = 'none'; /* Hide if image fails to load */ 
                        }}
                    />
                    <p className="text-xs text-center text-gray-500 mt-1">{altText}</p>
                </div>
            );
        };

        return (
            <div className="mt-6 space-y-6">
                <h3 className="text-lg font-medium text-gray-900 border-b pb-2">Visualizations</h3>

                {/* Overall Plot Section */}
                {plotPaths.overall && plotPaths.overall.temporal_distribution && (
                    <div>
                        <h4 className="text-md font-medium text-gray-700 mb-2">Overall Temporal Distribution</h4>
                        {createImage(plotPaths.overall.temporal_distribution, 'Overall Temporal Distribution Plot')}
                    </div>
                )}

                {/* User Plots Section */}
                {plotPaths.user_plots && Object.keys(plotPaths.user_plots).length > 0 && (
                    <div>
                        <h4 className="text-md font-medium text-gray-700 mb-3">User-Specific Plots</h4>
                        <div className="space-y-4">
                            {Object.entries(plotPaths.user_plots).map(([userId, userPlots]) => (
                                <div key={userId} className="p-4 border border-gray-200 rounded-lg bg-gray-50">
                                    <h5 className="text-sm font-semibold text-gray-800 mb-3">User: {userId}</h5>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        {createImage(userPlots.entity_distribution, `Entity Distribution for ${userId}`)}
                                        {createImage(userPlots.risk_profile, `Risk Profile for ${userId}`)}
                                        {createImage(userPlots.risk_trend, `Risk Trend for ${userId}`)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Fallback if no plots were generated but no error either */}
                {!(plotPaths.overall?.temporal_distribution) && !(plotPaths.user_plots && Object.keys(plotPaths.user_plots).length > 0) && (
                    <p className="text-sm text-gray-500">No specific plots were generated for this file.</p>
                )}
            </div>
        );
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
            <div className="mt-6 space-y-4">
                <div className="p-4 bg-gray-50 rounded-md border">
                    <h3 className="font-medium text-gray-900 mb-2">Analysis Summary</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>File Type: <span className="font-medium">{results.file_type.toUpperCase()}</span></div>
                        <div>Total Rows: <span className="font-medium">{results.total_rows}</span></div>
                        <div>Successful: <span className="font-medium text-green-600">{results.successful_rows}</span></div>
                        <div>Failed: <span className="font-medium text-red-600">{results.failed_rows}</span></div>
                        <div>Processing Time: <span className="font-medium">{(results.processing_time_ms / 1000).toFixed(2)}s</span></div>
                    </div>
                </div>

                {getClassificationSummary() && (
                    <div className="p-4 bg-white border rounded-md">
                        <h3 className="font-medium text-gray-900 mb-2">Classification Summary</h3>
                        <div className="flex flex-wrap gap-2">
                            <div className="px-3 py-1.5 bg-red-50 border border-red-100 rounded-md">
                                <span className="text-sm font-medium text-red-800">Malicious: </span>
                                <span className="text-sm">{getClassificationSummary().maliciousCount}</span>
                            </div>
                            <div className="px-3 py-1.5 bg-green-50 border border-green-100 rounded-md">
                                <span className="text-sm font-medium text-green-800">Non-Malicious: </span>
                                <span className="text-sm">{getClassificationSummary().nonMaliciousCount}</span>
                            </div>
                            {getClassificationSummary().unclassifiedCount > 0 && (
                                <div className="px-3 py-1.5 bg-gray-50 border border-gray-100 rounded-md">
                                    <span className="text-sm font-medium text-gray-800">Unclassified: </span>
                                    <span className="text-sm">{getClassificationSummary().unclassifiedCount}</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <div className="border rounded-md overflow-hidden">
                    <button
                        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 focus:outline-none transition-colors"
                        onClick={toggleResultsExpanded}
                    >
                        <h3 className="font-medium text-gray-900">Row-by-Row Results</h3>
                        <div className="flex items-center text-sm text-gray-500">
                            <span className="mr-2">{resultsExpanded ? 'Collapse' : 'Expand'}</span>
                            {resultsExpanded ? (
                                <ChevronUpIcon className="h-4 w-4" />
                            ) : (
                                <ChevronDownIcon className="h-4 w-4" />
                            )}
                        </div>
                    </button>
                    
                    {resultsExpanded && (
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Row</th>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Text</th>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Classification</th>
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
                                                    <span className="text-gray-400">N/A</span>
                                                ) : row.predicted_class ? (
                                                    <div>
                                                        <span 
                                                            className={`px-2 py-1 rounded-full text-xs font-semibold ${
                                                                row.predicted_class === 'malicious' 
                                                                ? 'bg-red-100 text-red-800' 
                                                                : 'bg-green-100 text-green-800'
                                                            }`}
                                                        >
                                                            {row.predicted_class === 'malicious' ? 'Malicious' : 'Non-Malicious'}
                                                        </span>
                                                        {row.malicious_probability !== null && (
                                                            <div className="text-xs text-gray-500 mt-1">
                                                                {(row.malicious_probability * 100).toFixed(1)}%
                                                            </div>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <span className="text-gray-400">Unclassified</span>
                                                )}
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
                                                    <summary className="text-blue-600 hover:text-blue-800 text-xs">View analysis</summary>
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
                    )}
                    
                    {!resultsExpanded && (
                        <div className="p-4 text-sm text-gray-500">
                            <p>The file contains {results.results.length} analyzed rows. Click to expand and view details.</p>
                        </div>
                    )}
                </div>
            </div>
        )}

        {renderPlots()}

        {/* Full Screen Image Modal */}
        {fullScreenImage && (
            <div 
                className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50"
                onClick={() => setFullScreenImage(null)}
            >
                <div className="relative max-w-[90vw] max-h-[90vh]">
                    <button 
                        className="absolute top-4 right-4 bg-black bg-opacity-50 text-white rounded-full p-2 hover:bg-opacity-70"
                        onClick={(e) => {
                            e.stopPropagation();
                            setFullScreenImage(null);
                        }}
                    >
                        <XMarkIcon className="h-6 w-6" />
                    </button>
                    <img 
                        src={fullScreenImage.url} 
                        alt={fullScreenImage.altText}
                        className="max-h-[90vh] max-w-[90vw] object-contain"
                        onClick={(e) => e.stopPropagation()}
                    />
                    <p className="text-center text-white mt-2">{fullScreenImage.altText}</p>
                </div>
            </div>
        )}
        </div>
    );
};

export default FileUploader; 