import streamlit as st
from PIL import Image
import os

# Set the full directory path where your images are stored
image_dir = 'D:/Projects/! Obsidian/3-Apr-2024/Remote/Noctal/charts'

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
if os.path.exists(image_dir):
    print("Directory exists")
else:
    print("Directory does not exist")

# Display images
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Checking image path: {image_path}")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=image_file, use_container_width=True)
    else:
        st.write(f"File not found: {image_path}")