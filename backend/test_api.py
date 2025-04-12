#!/usr/bin/env python3
"""
Test script for the Insider Threat Detection API

This script sends sample requests to the API to verify it's working correctly.
"""

import requests
import json
import sys
import time

# API endpoint
BASE_URL = "http://localhost:8000"

def test_system_info():
    """Test the system info endpoint"""
    print("Testing system info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        
        data = response.json()
        print("System Info:")
        print(f"  Version: {data['version']}")
        print(f"  Platform: {data['system_info']['platform']}")
        print(f"  Device: {data['system_info']['device']}")
        print(f"  Thread Pool Size: {data['system_info']['thread_pool_size']}")
        print(f"  Model Path: {data['model_path']}")
        print(f"  Status: {data['status']}")
        print("✅ System info endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ System info endpoint test failed: {e}")
        return False

def test_threat_detection(text):
    """Test the threat detection endpoint"""
    print(f"\nTesting threat detection endpoint with: '{text}'")
    
    try:
        response = requests.post(
            f"{BASE_URL}/detect-threats",
            json={"text": text, "timestamp": "2023-11-15 19:30:00"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        print("\nDetected Entities:")
        for entity in data["entities"]:
            print(f"  - {entity['word']} ({entity['entity_group']}, Score: {entity['score']:.4f})")
        
        print("\nAnalysis Result:")
        print(data["analysis_result"])
        
        print("✅ Threat detection endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Threat detection endpoint test failed: {e}")
        return False

def main():
    print("🔍 Starting API tests...\n")
    
    # Test system info
    system_info_success = test_system_info()
    
    # Test threat detection with sample messages
    sample_texts = [
        "I'm going to download the customer database tonight after everyone has gone home.",
        "Just pulled all the patient records from Phoenix Memorial's database. Working late tonight.",
        "The meeting is scheduled for 2pm in the conference room."
    ]
    
    # Only test threat detection if system info test passed
    if system_info_success:
        for text in sample_texts:
            test_threat_detection(text)
            time.sleep(1)  # Add a small delay between requests
    
    print("\n🏁 API tests completed")

if __name__ == "__main__":
    main() 