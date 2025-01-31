import streamlit as st
from PIL import Image
import os
from pathlib import Path
import subprocess
import sys
import pkg_resources

# Set the full directory path where your images are stored
charts_dir = Path("charts")

# Create the directory if it doesn't exist
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)
    st.write("Directory was missing. Created the directory.")
else:
    st.write("Directory exists.")

# Run the script to generate charts
try:
    subprocess.run(["python", "sprint_progress_chart.py"], check=True)
    st.success("Charts generated successfully!")
except subprocess.CalledProcessError as e:
    st.error(f"Failed to generate charts. Error: {e}")
except FileNotFoundError:
    st.error("'sprint_progress_chart.py' script not found!")

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

# Display images
for image_file in image_files:
    image_path = charts_dir / image_file
    print(f"Checking image path: {image_path}")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=image_file, use_container_width=True)
    else:
        st.write(f"File not found: {image_path}")

# Debugging: List installed packages
installed_packages = [dist.project_name for dist in pkg_resources.working_set]
st.write("Installed Packages:", installed_packages)

# Fallback installation for matplotlib
try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])