import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

class JSONLAnalyzer:
    def __init__(self, directory_path):
        """
        Initialize the analyzer with the directory containing JSONL files.
        
        Args:
            directory_path: Path to directory containing CC-MAIN-*.jsonl files
        """
        self.directory_path = Path(directory_path)
        self.stats = []
        
    def extract_file_info(self, filename):
        """Extract year and part number from filename."""
        pattern = r'CC-MAIN-(\d{4})-\d+-part-(\d+)\.jsonl'
        match = re.match(pattern, filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def estimate_tokens(self, text):
        """
        Estimate token count. Rule of thumb: 1 token ≈ 4 characters.
        This is an approximation commonly used for English text.
        """
        if not text:
            return 0
        
        # Method 2: Character-based (1 token ≈ 4 characters)
        char_count = len(text)
        token_estimate_chars = char_count // 4
        
        # Average both methods
        return token_estimate_chars
    
    def analyze_file(self, filepath):
        """Analyze a single JSONL file."""
        filename = filepath.name
        year, part = self.extract_file_info(filename)
        
        if year is None:
            print(f"Skipping {filename} - doesn't match expected pattern")
            return None
        
        total_tokens = 0
        total_chars = 0
        total_words = 0
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    try:
                        data = json.loads(line)
                        # Extract all text content from the JSON
                        text_content = self.extract_text_from_json(data)
                        
                        tokens = self.estimate_tokens(text_content)
                        total_tokens += tokens
                        total_chars += len(text_content)
                        total_words += len(text_content.split())
                    except json.JSONDecodeError:
                        continue
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            return {
                'filename': filename,
                'year': year,
                'part': part,
                'line_count': line_count,
                'total_tokens': total_tokens,
                'total_words': total_words,
                'total_chars': total_chars,
                'file_size_mb': file_size_mb,
                'tokens_per_line': total_tokens / line_count if line_count > 0 else 0,
                'words_per_token': total_words / total_tokens if total_tokens > 0 else 0,
            }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None
    
    def extract_text_from_json(self, obj):
        """Recursively extract all text from JSON object."""
        text = []
        
        if isinstance(obj, dict):
            for value in obj.values():
                text.append(self.extract_text_from_json(value))
        elif isinstance(obj, list):
            for item in obj:
                text.append(self.extract_text_from_json(item))
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)
        
        return ' '.join(text)
    
    def analyze_all_files(self):
        """Analyze all matching JSONL files in the directory."""
        jsonl_files = list(self.directory_path.glob('CC-MAIN-*.jsonl'))
        
        if not jsonl_files:
            print(f"No matching files found in {self.directory_path}")
            return
        
        print(f"Found {len(jsonl_files)} files to analyze...")
        
        for filepath in tqdm(jsonl_files):
            result = self.analyze_file(filepath)
            if result:
                self.stats.append(result)
        
        print(f"\nAnalyzed {len(self.stats)} files successfully")
    
    def generate_report(self):
        """Generate and save statistics report."""
        if not self.stats:
            print("No statistics to report")
            return
        
        df = pd.DataFrame(self.stats)
        
        # Save detailed statistics to CSV
        df.to_csv('/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_statistics.csv', index=False)
        print("\nDetailed statistics saved to '/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_statistics.csv'")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\nTotal files analyzed: {len(df)}")
        print(f"Years covered: {sorted(df['year'].unique())}")
        print(f"\nTotal tokens across all files: {df['total_tokens'].sum():,}")
        print(f"Total words across all files: {df['total_words'].sum():,}")
        print(f"Total file size: {df['file_size_mb'].sum():.2f} MB")
        
        print("\n" + "-"*70)
        print("Per-file averages:")
        print(f"  Average tokens per file: {df['total_tokens'].mean():,.0f}")
        print(f"  Average words per file: {df['total_words'].mean():,.0f}")
        print(f"  Average lines per file: {df['line_count'].mean():,.0f}")
        print(f"  Average words per token: {df['words_per_token'].mean():.3f}")
        
        # Statistics by year
        print("\n" + "-"*70)
        print("Statistics by year:")
        yearly = df.groupby('year').agg({
            'total_tokens': 'sum',
            'total_words': 'sum',
            'line_count': 'sum',
            'file_size_mb': 'sum',
            'filename': 'count'
        }).rename(columns={'filename': 'file_count'})
        
        print(yearly.to_string())
    
    def create_visualizations(self):
        """Create and save visualization plots."""
        if not self.stats:
            print("No data to visualize")
            return
        
        df = pd.DataFrame(self.stats)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('JSONL File Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Tokens per year
        yearly_tokens = df.groupby('year')['total_tokens'].sum() / 1e6 # convert to million
        axes[0, 0].bar(yearly_tokens.index, yearly_tokens.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Total Tokens (x 1e6)')
        axes[0, 0].set_title('Total Tokens by Year')
        axes[0, 0].ticklabel_format(style='plain', axis='y')
        for i, v in enumerate(yearly_tokens.values):
            axes[0, 0].text(yearly_tokens.index[i], v, f'{v:,.0f}M', ha='center', va='bottom')
        
        # Plot 2: File count per year
        yearly_count = df.groupby('year').size()
        axes[0, 1].bar(yearly_count.index, yearly_count.values, color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Files')
        axes[0, 1].set_title('File Count by Year')
        for i, v in enumerate(yearly_count.values):
            axes[0, 1].text(yearly_count.index[i], v, str(v), ha='center', va='bottom')
        
        # Plot 3: Distribution of tokens per file
        total_tokens = df['total_tokens'] / 1e6 # convert to million
        axes[1, 0].hist(total_tokens, bins=30, color='green', alpha=0.6, edgecolor='black')
        axes[1, 0].set_xlabel('Tokens per File (x 1e6)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Tokens per File')
        axes[1, 0].ticklabel_format(style='plain', axis='x')
        
        # Plot 4: Words per token ratio by year
        yearly_ratio = df.groupby('year')['words_per_token'].mean()
        axes[1, 1].plot(yearly_ratio.index, yearly_ratio.values, marker='o', linewidth=2, 
                       markersize=8, color='purple')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Words per Token')
        axes[1, 1].set_title('Average Words per Token by Year')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to '/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_analysis.png'")
        plt.close()
        
        # Additional plot: Timeline view of all files
        if len(df) > 1:
            fig, ax = plt.subplots(figsize=(14, 6))
            scatter = ax.scatter(df['year'], df['part'], s=df['total_tokens']/1e9, 
                               c=df['total_tokens'], cmap='viridis', alpha=0.6, 
                               edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Part Number', fontsize=12)
            ax.set_title('File Distribution (size = tokens, color = token count)', fontsize=14)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Total Tokens', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_timeline.png', dpi=300, bbox_inches='tight')
            print("Timeline visualization saved to '/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_timeline.png'")
            plt.close()


def main():
    """Main execution function."""
    # Set your directory path here
    # directory_path = input("Enter the directory path containing JSONL files (or press Enter for current directory): ").strip()
    
    if len(sys.argv) < 2:
        directory_path = "."
    else:
        directory_path = sys.argv[1]
    
    analyzer = JSONLAnalyzer(directory_path)
    
    print("Starting analysis...")
    analyzer.analyze_all_files()
    
    if analyzer.stats:
        analyzer.generate_report()
        analyzer.create_visualizations()
        print("\n" + "="*70)
        print("Analysis complete!")
        print("Output files created:")
        print("  - file_statistics.csv (detailed statistics)")
        print("  - file_analysis.png (main visualizations)")
        print("  - file_timeline.png (timeline view)")
        print("="*70)
    else:
        print("No files were successfully analyzed.")


if __name__ == "__main__":
    main()