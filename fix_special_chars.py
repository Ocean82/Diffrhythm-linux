#!/usr/bin/env python3
"""
Fix special characters in Python files for Windows console compatibility
"""

import os
import sys

def fix_file(file_path):
    """Replace special Unicode characters with ASCII equivalents"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common Unicode characters
    replacements = {
        '✓': 'OK',
        '✗': 'ERROR', 
        '⚠': 'WARN',
        '✔': 'OK',
        '✘': 'ERROR',
        '⚠️': 'WARN',
        '✅': 'OK',
        '❌': 'ERROR',
        '⚠️': 'WARN',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        'ℹ': 'INFO',
        'ℕ': 'N',
        'ℚ': 'Q',
        'ℝ': 'R',
        'ℤ': 'Z',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'ζ': 'zeta',
        'η': 'eta',
        'θ': 'theta',
        'λ': 'lambda',
        'μ': 'mu',
        'ξ': 'xi',
        'π': 'pi',
        'ρ': 'rho',
        'σ': 'sigma',
        'τ': 'tau',
        'φ': 'phi',
        'χ': 'chi',
        'ψ': 'psi',
        'ω': 'omega'
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Replace any remaining non-ASCII characters with '?'
    # This is a last resort to prevent encoding errors
    clean_content = []
    for c in content:
        if ord(c) < 128:
            clean_content.append(c)
        else:
            clean_content.append('?')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(clean_content))
    
    print(f"Fixed special characters in {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_special_chars.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    if os.path.exists(file_path):
        fix_file(file_path)
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
