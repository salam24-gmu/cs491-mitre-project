import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline';

const ThreatDetectionForm = () => {
    const [formData, setFormData] = useState({
        text: '',
        user_id: '',
        timestamp: '',
        tweet_id: ''
    });
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [activeTab, setActiveTab] = useState('single'); // 'single' or 'batch'
    
    // For batch analysis
    const [batchTexts, setBatchTexts] = useState([
        { text: '', timestamp: '', user_id: '', tweet_id: '' }
    ]);
    
    // Single text handlers
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevData => ({
        ...prevData,
        [name]: value
        }));
    };
    
    // Batch text handlers
    const handleBatchTextChange = (index, field, value) => {
        const updatedTexts = [...batchTexts];
        updatedTexts[index][field] = value;
        setBatchTexts(updatedTexts);
    };
    
    const addBatchText = () => {
        setBatchTexts([...batchTexts, { text: '', timestamp: '', user_id: '', tweet_id: '' }]);
    };
    
    const removeBatchText = (index) => {
        if (batchTexts.length === 1) {
        toast.error('You need at least one text entry');
        return;
        }
        
        const updatedTexts = batchTexts.filter((_, i) => i !== index);
        setBatchTexts(updatedTexts);
    };
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        // Validation
        if (activeTab === 'single') {
            if (!formData.text.trim()) {
                toast.error('Please enter some text for analysis');
                return;
            }
        } else { // 'batch'
            // Check if at least one text is non-empty
            const hasText = batchTexts.some(item => item.text.trim() !== '');
            if (!hasText) {
                toast.error('Please enter at least one text for analysis');
                return;
            }
        }
        
        setIsLoading(true);
        setResult(null);
        
        try {
            let response;
            
            if (activeTab === 'single') {
                response = await axios.post('/api/analyze-risk', {
                    text: formData.text,
                    user_id: formData.user_id || undefined,
                    timestamp: formData.timestamp || undefined,
                    tweet_id: formData.tweet_id || undefined
                });
            } else { // 'batch'
                // Filter out empty texts
                const textsToSend = batchTexts
                    .filter(item => item.text.trim() !== '')
                    .map(item => ({
                        text: item.text,
                        user_id: item.user_id || undefined,
                        timestamp: item.timestamp || undefined,
                        tweet_id: item.tweet_id || undefined
                    }));
                
                response = await axios.post('/api/batch-analyze-risk', {
                    texts: textsToSend
                });
            }
            
            setResult(response.data);
            toast.success('Analysis completed');
        } catch (error) {
            console.error('Analysis error:', error);
            toast.error(error.response?.data?.detail || 'Error analyzing text');
        } finally {
            setIsLoading(false);
        }
    };
    
    const renderSingleForm = () => {
        return (
            <div>
                <div className="mb-4">
                    <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-1">
                        Text to Analyze *
                    </label>
                    <textarea
                        id="text"
                        name="text"
                        rows={4}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Enter text to analyze for potential threats..."
                        value={formData.text}
                        onChange={handleInputChange}
                        required
                    />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div>
                        <label htmlFor="user_id" className="block text-sm font-medium text-gray-700 mb-1">
                            User ID (optional)
                        </label>
                        <input
                            type="text"
                            id="user_id"
                            name="user_id"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="user_123"
                            value={formData.user_id}
                            onChange={handleInputChange}
                        />
                    </div>
                    
                    <div>
                        <label htmlFor="timestamp" className="block text-sm font-medium text-gray-700 mb-1">
                            Timestamp (optional)
                        </label>
                        <input
                            type="text"
                            id="timestamp"
                            name="timestamp"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="2023-04-15T22:10:00Z"
                            value={formData.timestamp}
                            onChange={handleInputChange}
                        />
                    </div>
                    
                    <div>
                        <label htmlFor="tweet_id" className="block text-sm font-medium text-gray-700 mb-1">
                            Tweet/Post ID (optional)
                        </label>
                        <input
                            type="text"
                            id="tweet_id"
                            name="tweet_id"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            placeholder="tweet_456"
                            value={formData.tweet_id}
                            onChange={handleInputChange}
                        />
                    </div>
                </div>
            </div>
        );
    };
    
    // Render batch input fields
    const renderBatchForm = () => {
        return (
            <div className="space-y-4">
                {batchTexts.map((item, index) => (
                    <div key={index} className="p-4 border border-gray-200 rounded-md bg-gray-50">
                        <div className="flex justify-between items-center mb-3">
                            <h4 className="text-sm font-medium text-gray-700">Text Entry #{index + 1}</h4>
                            <button
                                type="button"
                                onClick={() => removeBatchText(index)}
                                className="text-red-500 hover:text-red-700"
                                aria-label="Remove text"
                            >
                                <TrashIcon className="h-4 w-4" />
                            </button>
                        </div>
                        
                        <div className="mb-3">
                            <label htmlFor={`text-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                                Text to Analyze *
                            </label>
                            <textarea
                                id={`text-${index}`}
                                rows={2}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                placeholder="Enter text to analyze..."
                                value={item.text}
                                onChange={(e) => handleBatchTextChange(index, 'text', e.target.value)}
                            />
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            <div>
                                <label htmlFor={`user-id-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                                    User ID
                                </label>
                                <input
                                    type="text"
                                    id={`user-id-${index}`}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="user_123"
                                    value={item.user_id}
                                    onChange={(e) => handleBatchTextChange(index, 'user_id', e.target.value)}
                                />
                            </div>
                            
                            <div>
                                <label htmlFor={`timestamp-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                                    Timestamp
                                </label>
                                <input
                                    type="text"
                                    id={`timestamp-${index}`}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="2023-04-15T22:10:00Z"
                                    value={item.timestamp}
                                    onChange={(e) => handleBatchTextChange(index, 'timestamp', e.target.value)}
                                />
                            </div>
                            
                            <div>
                                <label htmlFor={`tweet-id-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                                    Tweet/Post ID
                                </label>
                                <input
                                    type="text"
                                    id={`tweet-id-${index}`}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="tweet_456"
                                    value={item.tweet_id}
                                    onChange={(e) => handleBatchTextChange(index, 'tweet_id', e.target.value)}
                                />
                            </div>
                        </div>
                    </div>
                ))}
                
                <div className="flex justify-center">
                    <button
                        type="button"
                        onClick={addBatchText}
                        className="flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <PlusIcon className="h-4 w-4 mr-2" />
                        Add Another Text
                    </button>
                </div>
            </div>
        );
    };
    
    const renderResults = () => {
        if (!result) return null;
        
        return (
            <div className="mt-4 p-4 bg-gray-50 rounded-md">
                <h3 className="font-medium text-gray-900 mb-2">Risk Analysis Results</h3>
                
                {result.risk_level && (
                    <div className="mb-3">
                        <p className="text-sm font-medium">
                            Risk Level: 
                            <span 
                                className={`ml-2 px-2 py-1 rounded-full text-xs ${
                                result.risk_level === 'HIGH' ? 'bg-red-100 text-red-800' :
                                result.risk_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-green-100 text-green-800'
                                }`}
                            >
                                {result.risk_level}
                            </span>
                        </p>
                    </div>
                )}
                
                <div className="mt-3">
                    <details open>
                        <summary className="text-sm font-medium cursor-pointer text-blue-600 hover:text-blue-800">
                            View Details
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-x-auto max-h-60">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </details>
                </div>
                
                {result.processing_time_ms && (
                    <p className="text-xs text-gray-500 mt-3">
                        Processed in {(result.processing_time_ms / 1000).toFixed(3)} seconds
                    </p>
                )}
            </div>
        );
    };
    
    return (
        <div className="rounded-lg bg-white shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Threat Detection</h2>
            
            <div className="flex border-b mb-4">
                <button
                    className={`py-2 px-4 font-medium text-sm border-b-2 ${
                        activeTab === 'single' 
                        ? 'border-blue-500 text-blue-600' 
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                    onClick={() => setActiveTab('single')}
                >
                    Single Analysis
                </button>
                <button
                    className={`py-2 px-4 font-medium text-sm border-b-2 ${
                        activeTab === 'batch' 
                        ? 'border-blue-500 text-blue-600' 
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                    onClick={() => setActiveTab('batch')}
                >
                    Batch Analysis
                </button>
            </div>
            
            <form onSubmit={handleSubmit}>
                {activeTab === 'single' ? renderSingleForm() : renderBatchForm()}
                
                <div className="mt-4">
                    <button
                        type="submit"
                        disabled={isLoading}
                        className={`w-full py-2 px-4 rounded-md text-white font-medium ${
                        isLoading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
                        }`}
                    >
                        {isLoading ? "Analyzing..." : activeTab === 'single' ? "Analyze Text" : "Analyze Batch Texts"}
                    </button>
                </div>
            </form>
            
            {renderResults()}
        </div>
    );
};

export default ThreatDetectionForm; 