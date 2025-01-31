import streamlit as st
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import distributions

# Debug: List installed packages
installed_packages = [dist.metadata["Name"] for dist in distributions()]
st.write("Python executable being used: ", sys.executable)
st.write("Python paths: ", sys.path)
st.write("Installed Packages:", installed_packages)

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
    subprocess.run([sys.executable, "sprint_progress_chart.py"], check=True)
    st.success("Charts generated successfully!")
except subprocess.CalledProcessError as e:
    st.error(f"Failed to generate charts. Error: {e}")
except FileNotFoundError:
    st.error("'sprint_progress_chart.py' script not found!")

# Display all charts
chart_files = [
    "burndown_chart.png",
    "cycle_time_chart.png",
    "developer_focus.png",
    "developer_metrics.png",
    "developer_trends.png",
    "issue_distribution.png",
    "priority_distribution.png",
    "sprint_goals.png",
    "throughput_chart.png",
    "velocity_chart.png",
    "workload.png",
    "story_points_focus.png"
]

for chart_file in chart_files:
    chart_path = charts_dir / chart_file
    st.write(f"Checking image path: {chart_path}")
    if chart_path.exists():
        st.image(str(chart_path), caption=chart_file.replace(".png", "").replace("_", " ").title())
    else:
        st.warning(f"Chart not found: {chart_file}")