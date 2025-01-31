import streamlit as st
from PIL import Image
import os
from pathlib import Path
import subprocess

# Set the full directory path where your images are stored
charts_dir = Path("charts")

# Run the script to generate charts
try:
    subprocess.run(["python", "sprint_progress_chart.py"], check=True)
except subprocess.CalledProcessError as e:
    st.error(f"Error generating charts: {e}")

# List of image files with standardized names
image_files = [
    'burndown_chart.png',
    'cycle_time_chart.png',
    'developer_focus.png',
    'developer_metrics.png',
    'developer_trends.png',
    'issue_distribution.png',
    'priority_distribution.png',
    'sprint_goals.png',
    'throughput_chart.png',
    'velocity_chart.png',
    'workload.png',
    'story_points_focus.png'
]

# Streamlit app
st.title("Interactive Dashboard")

# Check if the directory exists
if os.path.exists(charts_dir):
    print("Directory exists")
else:
    print("Directory does not exist")

# Display images
for image_file in image_files:
    image_path = charts_dir / image_file
    print(f"Checking image path: {image_path}")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=image_file, use_container_width=True)
    else:
        st.write(f"File not found: {image_path}")