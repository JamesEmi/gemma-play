#!/usr/bin/env python3

import os
import numpy as np
import wave
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_audio_data(npy_file):
    """Analyze a single npy file and return statistics"""
    data = np.load(npy_file)
    
    stats = {
        'filename': os.path.basename(npy_file),
        'length': len(data),
        'duration_44k': len(data) / 44100,  # assuming 44.1kHz
        'duration_48k': len(data) / 48000,  # assuming 48kHz
        'duration_16k': len(data) / 16000,  # assuming 16kHz
        'min_val': np.min(data),
        'max_val': np.max(data),
        'mean_val': np.mean(data),
        'std_val': np.std(data),
        'abs_max': np.max(np.abs(data))
    }
    
    return data, stats

def convert_to_wav(data, output_path, sample_rate=44100, normalize=True):
    """Convert numpy array to WAV file"""
    
    # Normalize data if requested
    if normalize and np.max(np.abs(data)) > 0:
        # Normalize to prevent clipping, leaving some headroom
        data = data / np.max(np.abs(data)) * 0.95
    
    # Convert to 16-bit PCM
    data_int16 = (data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data_int16.tobytes())

def plot_waveform(data, title, save_path=None):
    """Plot waveform for visualization"""
    plt.figure(figsize=(12, 4))
    time_axis = np.arange(len(data)) / 44100  # assuming 44.1kHz for time axis
    plt.plot(time_axis, data)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze and convert NPY audio files to WAV')
    parser.add_argument('input_dir', help='Directory containing .npy files')
    parser.add_argument('--output_dir', help='Output directory for WAV files (default: input_dir/wav_output)')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate for WAV files (default: 44100)')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze files, don\'t convert to WAV')
    parser.add_argument('--plot_samples', type=int, default=0, help='Number of sample waveforms to plot (default: 0)')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize audio data (default: True)')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false', help='Don\'t normalize audio data')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'wav_output'
    
    if not args.analyze_only:
        output_dir.mkdir(exist_ok=True)
        plot_dir = output_dir / 'plots'
        if args.plot_samples > 0:
            plot_dir.mkdir(exist_ok=True)
    
    # Find all .npy files
    npy_files = list(input_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    print("-" * 80)
    
    # Analyze all files
    all_stats = []
    for i, npy_file in enumerate(sorted(npy_files)):
        try:
            data, stats = analyze_audio_data(npy_file)
            all_stats.append(stats)
            
            # Print statistics
            print(f"File: {stats['filename']}")
            print(f"  Samples: {stats['length']}")
            print(f"  Duration @ 44.1kHz: {stats['duration_44k']:.3f}s")
            print(f"  Duration @ 48kHz: {stats['duration_48k']:.3f}s") 
            print(f"  Duration @ 16kHz: {stats['duration_16k']:.3f}s")
            print(f"  Range: [{stats['min_val']:.6f}, {stats['max_val']:.6f}]")
            print(f"  Mean: {stats['mean_val']:.6f}, Std: {stats['std_val']:.6f}")
            print(f"  Max absolute: {stats['abs_max']:.6f}")
            
            if not args.analyze_only:
                # Convert to WAV
                wav_filename = npy_file.stem + '.wav'
                wav_path = output_dir / wav_filename
                convert_to_wav(data, wav_path, args.sample_rate, args.normalize)
                print(f"  -> Converted to {wav_path}")
                
                # Plot sample waveforms
                if args.plot_samples > 0 and i < args.plot_samples:
                    plot_path = plot_dir / (npy_file.stem + '_waveform.png')
                    plot_title = f"Waveform: {stats['filename']} ({stats['length']} samples, {stats['duration_44k']:.3f}s @ 44.1kHz)"
                    plot_waveform(data, plot_title, plot_path)
                    print(f"  -> Plot saved to {plot_path}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
            continue
    
    # Summary statistics
    if all_stats:
        lengths = [s['length'] for s in all_stats]
        durations_44k = [s['duration_44k'] for s in all_stats]
        abs_maxes = [s['abs_max'] for s in all_stats]
        
        print(f"\nSUMMARY:")
        print(f"Total files processed: {len(all_stats)}")
        print(f"Sample counts - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
        print(f"Durations @ 44.1kHz - Min: {min(durations_44k):.3f}s, Max: {max(durations_44k):.3f}s, Mean: {np.mean(durations_44k):.3f}s")
        print(f"Max absolute values - Min: {min(abs_maxes):.6f}, Max: {max(abs_maxes):.6f}, Mean: {np.mean(abs_maxes):.6f}")
        
        if not args.analyze_only:
            print(f"\nWAV files saved to: {output_dir}")
            print(f"Sample rate: {args.sample_rate} Hz")
            print(f"Normalization: {'Enabled' if args.normalize else 'Disabled'}")

if __name__ == "__main__":
    main()