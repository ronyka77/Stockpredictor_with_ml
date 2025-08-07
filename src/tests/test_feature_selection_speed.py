#!/usr/bin/env python3
"""
Test script to demonstrate speed improvements in feature selection
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.time_series.mlp_main import MLPPredictorWithMLflow
    
    print("✅ Successfully imported MLPPredictorWithMLflow")
    
    # Create sample data for testing
    print("\n📊 Creating sample data for testing...")
    np.random.seed(42)
    
    # Create a realistic dataset
    n_samples = 1000
    n_features = 200
    
    # Generate features with some correlation to target
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create target with some correlation to features
    y = (X.iloc[:, :10].sum(axis=1) * 0.3 + 
         X.iloc[:, 10:20].sum(axis=1) * 0.2 + 
         np.random.randn(n_samples) * 0.1)
    
    print(f"   Created dataset: {n_samples} samples, {n_features} features")
    
    # Test different feature selection methods
    print("\n🚀 Testing feature selection methods...")
    
    # Create model instance
    model = MLPPredictorWithMLflow(
        model_name="test_feature_selector",
        config={'input_size': n_features}
    )
    
    # Test fast correlation method
    print("\n1. Testing fast correlation method...")
    start_time = time.time()
    try:
        features_corr = model.select_features_fast(X, y, n_features_to_select=30, method='correlation')
        corr_time = time.time() - start_time
        print(f"   ✅ Correlation method: {corr_time:.2f}s, selected {len(features_corr)} features")
    except Exception as e:
        print(f"   ❌ Correlation method failed: {e}")
        corr_time = float('inf')
    
    # Test fast variance method
    print("\n2. Testing fast variance method...")
    start_time = time.time()
    try:
        features_var = model.select_features_fast(X, y, n_features_to_select=30, method='variance')
        var_time = time.time() - start_time
        print(f"   ✅ Variance method: {var_time:.2f}s, selected {len(features_var)} features")
    except Exception as e:
        print(f"   ❌ Variance method failed: {e}")
        var_time = float('inf')
    
    # Test lightweight MLP method
    print("\n3. Testing lightweight MLP method...")
    start_time = time.time()
    try:
        features_mlp = model.select_features_fast(X, y, n_features_to_select=30, method='lightweight_mlp')
        mlp_time = time.time() - start_time
        print(f"   ✅ Lightweight MLP method: {mlp_time:.2f}s, selected {len(features_mlp)} features")
    except Exception as e:
        print(f"   ❌ Lightweight MLP method failed: {e}")
        mlp_time = float('inf')
    
    # Test original detailed method
    print("\n4. Testing original detailed method...")
    start_time = time.time()
    try:
        features_detailed = model.select_features(X, y, n_features_to_select=30, use_fast=False)
        detailed_time = time.time() - start_time
        print(f"   ✅ Detailed method: {detailed_time:.2f}s, selected {len(features_detailed)} features")
    except Exception as e:
        print(f"   ❌ Detailed method failed: {e}")
        detailed_time = float('inf')
    
    # Summary
    print("\n📈 Performance Summary:")
    methods = [
        ('Correlation', corr_time),
        ('Variance', var_time),
        ('Lightweight MLP', mlp_time),
        ('Detailed MLP', detailed_time)
    ]
    
    successful_methods = [(name, time) for name, time in methods if time != float('inf')]
    
    if successful_methods:
        fastest = min(successful_methods, key=lambda x: x[1])
        slowest = max(successful_methods, key=lambda x: x[1])
        
        print(f"   🏆 Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
        print(f"   🐌 Slowest: {slowest[0]} ({slowest[1]:.2f}s)")
        
        if slowest[1] > 0:
            speedup = slowest[1] / fastest[1]
            print(f"   ⚡ Speedup: {speedup:.1f}x faster")
    
    print("\n✅ Feature selection speed test completed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 