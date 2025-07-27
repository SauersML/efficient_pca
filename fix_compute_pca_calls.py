#!/usr/bin/env python3
import re
import sys

def fix_compute_pca_calls(content):
    """Fix compute_pca calls by adding the missing snp_metadata parameter"""
    
    # Pattern to match compute_pca calls with only 2 arguments
    # This matches: .compute_pca(&something, &something_else)
    pattern = r'(\.compute_pca\([^,\)]+,\s*[^,\)]+)\)'
    
    # For each match, we need to insert the third argument
    def replacement(match):
        # Get the existing part with 2 args
        prefix = match.group(1)
        # Add the third argument 
        return prefix + ', &snp_metadata)'
    
    # Apply the replacement
    result = re.sub(pattern, replacement, content)
    
    return result

def add_snp_metadata_creation(content):
    """Add snp_metadata creation before compute_pca calls"""
    
    # Pattern to find places where we need to add snp_metadata creation
    # Look for compute_pca calls and add the metadata creation before them
    lines = content.split('\n')
    result_lines = []
    
    for i, line in enumerate(lines):
        # Check if this line contains compute_pca and doesn't already have snp_metadata creation nearby
        if '.compute_pca(' in line and '&snp_metadata' in line:
            # Check if the previous few lines already have snp_metadata creation
            has_metadata = False
            for j in range(max(0, i-5), i):
                if 'snp_metadata = create_dummy_snp_metadata' in lines[j]:
                    has_metadata = True
                    break
            
            if not has_metadata:
                # Find the number of SNPs - look for patterns like (0..num_snps) or similar
                metadata_line = None
                for j in range(max(0, i-10), i+1):
                    if 'map(PcaSnpId).collect()' in lines[j]:
                        # Extract the variable name (likely num_snps)
                        match = re.search(r'\(0\.\.(\w+)\)', lines[j])
                        if match:
                            var_name = match.group(1)
                            metadata_line = f'            let snp_metadata = create_dummy_snp_metadata({var_name});'
                            break
                
                if metadata_line:
                    # Add metadata creation line before the current line
                    result_lines.append(metadata_line)
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)

if __name__ == '__main__':
    # Read the file
    with open('tests/eigensnp_tests.rs', 'r') as f:
        content = f.read()
    
    # Fix the compute_pca calls
    content = fix_compute_pca_calls(content)
    content = add_snp_metadata_creation(content)
    
    # Write back
    with open('tests/eigensnp_tests.rs', 'w') as f:
        f.write(content)
    
    print("Fixed compute_pca calls in tests/eigensnp_tests.rs")
