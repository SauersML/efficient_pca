#!/usr/bin/env python3
import re

def fix_compute_pca_calls(content):
    """Fix compute_pca calls by adding the missing snp_metadata parameter"""
    
    # Pattern to match compute_pca calls with only 2 arguments
    pattern = r'(\.compute_pca\([^,\)]+,\s*[^,\)]+)\)'
    
    def replacement(match):
        prefix = match.group(1)
        return prefix + ', &snp_metadata)'
    
    result = re.sub(pattern, replacement, content)
    return result

def add_snp_metadata_creation(content):
    """Add snp_metadata creation before compute_pca calls"""
    
    lines = content.split('\n')
    result_lines = []
    
    for i, line in enumerate(lines):
        if '.compute_pca(' in line and '&snp_metadata' in line:
            # Check if the previous few lines already have snp_metadata creation
            has_metadata = False
            for j in range(max(0, i-5), i):
                if 'snp_metadata = ' in lines[j] and ('create_dummy_snp_metadata' in lines[j] or 'PcaSnpMetadata' in lines[j]):
                    has_metadata = True
                    break
            
            if not has_metadata:
                # For diagnostics tests, we need to create metadata based on the LD block specs
                # Look for ld_block_specs variable
                metadata_line = None
                for j in range(max(0, i-10), i+1):
                    if 'ld_block_specs' in lines[j] or 'LdBlockSpecification' in lines[j]:
                        # Add a generic metadata creation
                        metadata_line = '    let snp_metadata: Vec<efficient_pca::eigensnp::PcaSnpMetadata> = (0..mock_data_accessor.num_pca_snps()).map(|i| efficient_pca::eigensnp::PcaSnpMetadata { id: std::sync::Arc::new(format!("snp_{}", i)), chr: std::sync::Arc::new("chr1".to_string()), pos: i as u64 * 1000 + 100000 }).collect();'
                        break
                
                if metadata_line:
                    result_lines.append(metadata_line)
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)

if __name__ == '__main__':
    try:
        with open('tests/eigensnp_diagnostics_tests.rs', 'r') as f:
            content = f.read()
        
        content = fix_compute_pca_calls(content)
        content = add_snp_metadata_creation(content)
        
        with open('tests/eigensnp_diagnostics_tests.rs', 'w') as f:
            f.write(content)
        
        print("Fixed compute_pca calls in tests/eigensnp_diagnostics_tests.rs")
    except FileNotFoundError:
        print("tests/eigensnp_diagnostics_tests.rs not found, skipping")
