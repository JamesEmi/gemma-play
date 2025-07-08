#!/usr/bin/env python3
"""
Extract audio using ROS2 bag API - should work better with CDR encoding
"""

import os
import numpy as np
import argparse
from pathlib import Path

def extract_audio_ros2bag(mcap_file, output_dir, topic_name="/iphone3/audio_data"):
    """Extract audio using ros2bag API"""
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except ImportError as e:
        print(f"Error importing ROS2 packages: {e}")
        print("Make sure you're in a ROS2 environment and have sourced the workspace")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up bag reader
    storage_options = StorageOptions(uri=str(mcap_file), storage_id='mcap')
    converter_options = ConverterOptions('', '')
    
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag file: {e}")
        return False
    
    # Get message type
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    print("Available topics:")
    for topic, msg_type in type_map.items():
        print(f"  {topic}: {msg_type}")
    
    if topic_name not in type_map:
        print(f"\\nError: Topic {topic_name} not found in bag")
        return False
    
    msg_type_str = type_map[topic_name]
    print(f"\\nExtracting from topic: {topic_name}")
    print(f"Message type: {msg_type_str}")
    
    try:
        msg_type = get_message(msg_type_str)
    except Exception as e:
        print(f"Error getting message type for {msg_type_str}: {e}")
        print("Make sure the message type package is available in your environment")
        return False
    
    print(f"Output directory: {output_dir}")
    
    count = 0
    while reader.has_next():
        try:
            topic, data, timestamp = reader.read_next()
            
            if topic == topic_name:
                # Deserialize the message
                ros_msg = deserialize_message(data, msg_type)
                
                # Extract the audio data
                audio_data = np.array(ros_msg.data, dtype=np.float32)
                
                # Save as npy file with timestamp
                filename = f"{timestamp}.npy"
                filepath = output_dir / filename
                np.save(filepath, audio_data)
                
                print(f"Saved: {filename} ({len(audio_data)} samples)")
                count += 1
                
        except Exception as e:
            print(f"Error processing message: {e}")
            continue
    
    print(f"\\nExtracted {count} audio messages")
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract audio using ROS2 bag API')
    parser.add_argument('mcap_file', help='Path to MCAP file')
    parser.add_argument('output_dir', help='Output directory for audio files')
    parser.add_argument('--topic', default='/iphone3/audio_data', help='Audio topic name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mcap_file):
        print(f"Error: MCAP file {args.mcap_file} not found")
        return
    
    success = extract_audio_ros2bag(args.mcap_file, args.output_dir, args.topic)
    
    if success:
        print(f"\\nAudio files saved to: {args.output_dir}")
        print("Use the convert_npy_to_wav.py script to analyze and convert to WAV")

if __name__ == "__main__":
    main()
