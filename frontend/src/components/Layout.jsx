import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';

const Layout = ({ children }) => {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    
    return (
        <div className="min-h-screen bg-gray-100 flex flex-col">
        <Toaster position="top-right" />
        
        {/* Mobile sidebar */}
        <div 
            className={`fixed inset-0 z-40 lg:hidden ${sidebarOpen ? 'block' : 'hidden'}`}
            onClick={() => setSidebarOpen(false)}
        >
            <div className="fixed inset-0 bg-gray-600 bg-opacity-75" />
        </div>
        
        <div className="flex h-full">
            {/* Sidebar */}
            <div className={`
                fixed inset-y-0 left-0 flex flex-col z-40 w-64 bg-white border-r border-gray-200
                transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-auto
                ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
            `}>
                <div className="flex items-center justify-between h-16 flex-shrink-0 px-4 border-b">
                <h1 className="text-xl font-bold text-gray-900">MITRE Analyzer</h1>
                <button
                    className="lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                >
                    <XMarkIcon className="h-6 w-6" />
                </button>
                </div>
                
                <div className="overflow-y-auto flex-1 p-4">
                <nav className="space-y-1">
                    <a href="#threat-detection" className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-900 bg-gray-100">
                    Threat Detection
                    </a>
                    <a href="#file-analysis" className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50 hover:text-gray-900">
                    File Analysis
                    </a>
                </nav>
                </div>
            </div>
            
            {/* Main content area */}
            <div className="flex-1 flex flex-col lg:pl-0">
                {/* Top header */}
                <header className="bg-white shadow-sm">
                    <div className="flex h-16 items-center justify-between px-4">
                        <button
                            className="lg:hidden"
                            onClick={() => setSidebarOpen(true)}
                        >
                            <Bars3Icon className="h-6 w-6" />
                        </button>
                        
                        <div className="ml-auto">
                        </div>
                    </div>
                </header>
                
                {/* Main content */}
                <main className="flex-1 py-4 px-4 sm:px-6 lg:px-8 bg-gray-100">
                    {children}
                </main>
                
                {/* Footer */}
                <footer className="bg-white border-t border-gray-200 p-4 mt-auto">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col items-center justify-center text-center">
                    <p className="text-sm text-gray-500 mb-3">
                    © {new Date().getFullYear()} MITRE Analyzer. All rights reserved.
                    </p>
                    <a
                        href="https://github.com/MITRE/cs491-mitre-project"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-500 hover:text-gray-900"
                        aria-label="GitHub repository"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                        </svg>
                    </a>
                </div>
                </footer>
            </div>
        </div>
        </div>
    );
};

export default Layout; 