import os
import sys
import subprocess
import logging
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to ensure package is installed
def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Required packages
required_packages = ['matplotlib', 'numpy', 'pandas', 'seaborn']
for package in required_packages:
    ensure_package(package)

# Now import matplotlib and set backend
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jira import JIRA
from dotenv import load_dotenv
import streamlit as st
import argparse

# Load environment variables
load_dotenv()

# JIRA configuration
jira_url = os.getenv("JIRA_URL")
jira_username = os.getenv("JIRA_USERNAME")
jira_api_token = os.getenv("JIRA_API_TOKEN")

if not all([jira_url, jira_username, jira_api_token]):
    logger.error("Missing JIRA credentials. Please check your .env file")
    sys.exit(1)

# Set up JIRA client
options = {
    'server': jira_url,
    'verify': True
}

try:
    jira = JIRA(options, basic_auth=(jira_username, jira_api_token))
    logger.info("Successfully connected to JIRA")
except Exception as e:
    logger.error(f"Failed to connect to JIRA: {e}")
    sys.exit(1)

# Debug information
print("Python executable being used: ", sys.executable)
print("Python paths: ", sys.path)

# Debug to check if matplotlib is installed
installed_packages = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
print("Installed Packages: ")
print(installed_packages.stdout)

# Now import matplotlib and other packages
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jira import JIRA
from dotenv import load_dotenv
from collections import defaultdict
import streamlit as st
import argparse
import logging
import json

# Load environment variables from .env file
load_dotenv()

# Use secrets for authentication
try:
    jira_url = st.secrets["JIRA_URL"]
    jira_username = st.secrets["JIRA_USERNAME"]
    jira_api_token = st.secrets["JIRA_API_TOKEN"]
except Exception:
    jira_url = os.getenv("JIRA_URL")
    jira_username = os.getenv("JIRA_USERNAME")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

# Ensure essential environment variables are set
if not all([jira_url, jira_username, jira_api_token]):
    raise ValueError("Missing one or more required environment variables: JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN.")

options = {"server": jira_url}
jira = JIRA(options, basic_auth=(jira_username, jira_api_token))

# Setup logging
log = logging.getLogger('jira_metrics')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(logging.INFO)

# Default settings
DEFAULT_SERVER = os.getenv('JIRA_URL', 'https://noctal-team.atlassian.net')
DEFAULT_PROJECT = "SCRUM"
DEFAULT_TYPES = ['Bug', 'Story', 'Task']

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch Jira data and create productivity charts')
    parser.add_argument('-s', '--server', default=DEFAULT_SERVER, help='Jira server URL')
    parser.add_argument('-u', '--username', default=os.getenv('JIRA_USERNAME'), help='Jira username')
    parser.add_argument('-t', '--token', default=os.getenv('JIRA_API_TOKEN'), help='Jira API token')
    parser.add_argument('-p', '--project', default=DEFAULT_PROJECT, help='Jira project key')
    parser.add_argument('-b', '--begin_date', 
                       default=(datetime.date.today() - datetime.timedelta(days=30)).isoformat(),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('-e', '--end_date',
                       default=datetime.date.today().isoformat(),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--charts', nargs='+', 
                       default=['velocity', 'cycle_time', 'cycle_time_trend', 'issue_distribution', 
                               'burndown', 'throughput', 'workload', 'sprint_goals', 'priority',
                               'developer_metrics', 'developer_focus', 'developer_trends',
                               'story_points_focus'],
                       help='Types of charts to generate')
    parser.add_argument('--output-dir', default='Local Code/jira_analyzer/charts', help='Directory for output charts')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.username:
        parser.error("JIRA_USERNAME environment variable is not set")
    if not args.token:
        parser.error("JIRA_API_TOKEN environment variable is not set")
    
    return vars(args)

def fetch_jira_data(server, username, token, project, start_date, end_date):
    options = {"server": server}
    try:
        jira = JIRA(options, basic_auth=(username, token))
        jql_query = (f'project = {project} AND '
                    f'created >= "{start_date}" AND created <= "{end_date}" '
                    f'ORDER BY created DESC')
        
        log.info(f"Fetching Jira issues with query: {jql_query}")
        issues = jira.search_issues(jql_query, maxResults=1000, fields=[
            'summary', 'issuetype', 'status', 'created', 'updated',
            'assignee', 'priority', 'customfield_10016',
            'resolutiondate', 'components'
        ])
    except Exception as e:
        log.error(f"Failed to fetch Jira data: {e}")
        return []
    
    data = []
    for issue in issues:
        fields = issue.fields
        data.append({
            "Key": issue.key,
            "Summary": fields.summary,
            "Type": fields.issuetype.name,
            "Status": fields.status.name,
            "Created": fields.created[:10],
            "Updated": fields.updated[:10],
            "Resolution_Date": fields.resolutiondate[:10] if fields.resolutiondate else None,
            "Assignee": getattr(fields.assignee, 'displayName', 'Unassigned'),
            "Priority": fields.priority.name if fields.priority else 'None',
            "Story_Points": getattr(fields, 'customfield_10016', 0) or 0,
            "Components": [c.name for c in fields.components] if fields.components else []
        })
    return data

def plot_throughput(df, output_file):
    df['Created'] = pd.to_datetime(df['Created'])
    throughput = df.groupby(df['Created'].dt.date).size()
    
    plt.figure(figsize=(12, 6))
    throughput.plot(kind='bar', color='skyblue')
    plt.title("Issues Completed Per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Issues")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    log.info("Throughput chart saved.")

def plot_issue_distribution(df, output_file):
    plt.figure(figsize=(12, 6))
    df_pivot = df.pivot_table(index='Type', columns='Status', aggfunc='size', fill_value=0)
    df_pivot.plot(kind='bar', stacked=True)
    plt.title('Issue Distribution by Type and Status')
    plt.xlabel('Issue Type')
    plt.ylabel('Number of Issues')
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    log.info("Issue distribution chart saved.")

def create_cycle_time_chart(df, output_dir):
    """Create a clear and informative cycle time chart showing distribution by issue type."""
    # Filter for completed issues and calculate cycle time
    df_done = df[df['Status'] == 'Done'].copy()
    df_done['Created'] = pd.to_datetime(df_done['Created'])
    df_done['Resolution_Date'] = pd.to_datetime(df_done['Resolution_Date'])
    df_done['Cycle_Time'] = (df_done['Resolution_Date'] - df_done['Created']).dt.days

    # Check if we have enough data
    if len(df_done) == 0:
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No completed issues in the selected time period',
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(f"{output_dir}/cycle_time_chart.png")
        plt.close()
        log.warning("No completed issues found for cycle time chart")
        return

    # Create figure with more height for better readability
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn-v0_8')

    # Create violin plot with individual points
    sns.violinplot(data=df_done, x='Type', y='Cycle_Time', inner='box', color='lightblue')
    sns.stripplot(data=df_done, x='Type', y='Cycle_Time', color='red', 
                 size=8, alpha=0.5, jitter=0.2)

    # Calculate and display statistics
    stats = df_done.groupby('Type').agg({
        'Cycle_Time': ['count', 'mean', 'median', 'std', 
                      lambda x: x.quantile(0.85)]
    }).round(1)
    stats.columns = ['Count', 'Mean', 'Median', 'Std', '85th']

    # Add statistics as a table below the plot
    table_data = []
    table_columns = ['Issue Type', 'Count', 'Mean (days)', 'Median (days)', '85th %ile']
    for idx, row in stats.iterrows():
        table_data.append([
            idx,
            int(row['Count']),
            f"{row['Mean']:.1f}",
            f"{row['Median']:.1f}",
            f"{row['85th']:.1f}"
        ])

    # Create the table
    plt.table(cellText=table_data,
             colLabels=table_columns,
             cellLoc='center',
             loc='bottom',
             bbox=[0, -0.35, 1, 0.2])

    # Add overall statistics in top left
    overall_stats = (
        f"Overall Statistics:\n"
        f"Total Issues: {len(df_done)}\n"
        f"Avg Cycle Time: {df_done['Cycle_Time'].mean():.1f}d\n"
        f"Median Cycle Time: {df_done['Cycle_Time'].median():.1f}d"
    )
    plt.text(-0.2, plt.ylim()[1], overall_stats,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                      boxstyle='round,pad=0.5'),
             verticalalignment='top')

    # Customize the plot
    plt.title('Cycle Time Distribution by Issue Type\n(How long does it take to complete different types of work?)',
             pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Issue Type', fontsize=12, fontweight='bold')
    plt.ylabel('Days to Complete', fontsize=12, fontweight='bold')

    # Add explanatory text
    explanation = (
        'How to read this chart:\n'
        '• Blue shape: Distribution of completion times\n'
        '• White dot: Median completion time\n'
        '• Box: Shows the middle 50% of completion times\n'
        '• Red dots: Individual issues\n'
        '• Width: Density of values at each point'
    )
    plt.text(plt.xlim()[1], plt.ylim()[1], explanation,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                      boxstyle='round,pad=0.5'),
             horizontalalignment='right',
             verticalalignment='top')

    # Add grid and adjust layout
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom to make room for the table
    
    plt.savefig(f"{output_dir}/cycle_time_chart.png", bbox_inches='tight', dpi=300)
    plt.close()
    log.info("Cycle time chart saved.")

def create_velocity_trend_chart(df, output_dir):
    """Create a comprehensive velocity analysis with multiple views."""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Calculate weekly metrics for completed issues
    df_done = df[df['Status'] == 'Done'].copy()
    df_done['Created'] = pd.to_datetime(df_done['Created'])
    weekly_data = df_done.groupby(pd.Grouper(key='Created', freq='W')).agg({
        'Story_Points': 'sum',
        'Key': 'count'
    }).fillna(0)

    # If we have no data, show a message
    if len(weekly_data) == 0 or weekly_data.sum().sum() == 0:
        plt.text(0.5, 0.5, 'No completed issues found in the selected time period',
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(f"{output_dir}/velocity_chart.png")
        plt.close()
        log.warning("No velocity data to display")
        return

    # 1. Velocity Trends (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot story points
    color1 = '#2ecc71'  # Green
    ax1.set_ylabel('Story Points', color=color1)
    line1 = ax1.plot(weekly_data.index, weekly_data['Story_Points'], 
                     color=color1, marker='o', label='Story Points')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Add moving average for story points if we have enough data
    if len(weekly_data) >= 3:
        window = min(3, len(weekly_data))  # Use smaller window if less than 3 weeks
        sp_ma = weekly_data['Story_Points'].rolling(window=window).mean()
        ax1.plot(weekly_data.index, sp_ma, '--', color=color1, alpha=0.5, 
                label=f'{window}-Week SP Avg')
    
    # Plot number of issues on secondary axis
    ax1_twin = ax1.twinx()
    color2 = '#3498db'  # Blue
    ax1_twin.set_ylabel('Number of Issues', color=color2)
    line2 = ax1_twin.plot(weekly_data.index, weekly_data['Key'], 
                         color=color2, marker='s', label='Issues')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add moving average for issues if we have enough data
    if len(weekly_data) >= 3:
        issues_ma = weekly_data['Key'].rolling(window=window).mean()
        ax1_twin.plot(weekly_data.index, issues_ma, '--', color=color2, alpha=0.5, 
                     label=f'{window}-Week Issues Avg')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Weekly Velocity Trends')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # 2. Weekly Comparison (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create bar chart instead of violin plot
    x = np.arange(len(weekly_data))
    width = 0.35

    # Normalize the values to make them comparable
    max_sp = weekly_data['Story_Points'].max() if weekly_data['Story_Points'].max() > 0 else 1
    max_issues = weekly_data['Key'].max() if weekly_data['Key'].max() > 0 else 1
    normalized_sp = weekly_data['Story_Points'] / max_sp
    normalized_issues = weekly_data['Key'] / max_issues

    bars1 = ax2.bar(x - width/2, normalized_sp, width, label='Story Points', color=color1)
    bars2 = ax2.bar(x + width/2, normalized_issues, width, label='Issues', color=color2)

    # Add value labels
    def autolabel(bars, original_values):
        for bar, val in zip(bars, original_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(val)}',
                    ha='center', va='bottom')

    autolabel(bars1, weekly_data['Story_Points'])
    autolabel(bars2, weekly_data['Key'])

    ax2.set_title('Weekly Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in weekly_data.index], rotation=45)
    ax2.legend()
    ax2.set_ylabel('Normalized Values')
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative Progress (Bottom)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Calculate cumulative metrics
    cumulative_data = pd.DataFrame({
        'Story_Points': weekly_data['Story_Points'].cumsum(),
        'Issues': weekly_data['Key'].cumsum()
    }, index=weekly_data.index)
    
    # Plot cumulative progress
    ax3.plot(cumulative_data.index, cumulative_data['Story_Points'], 
             color=color1, marker='o', label='Story Points')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(cumulative_data.index, cumulative_data['Issues'], 
                 color=color2, marker='s', label='Issues')
    
    # Add labels
    ax3.set_ylabel('Cumulative Story Points', color=color1)
    ax3_twin.set_ylabel('Cumulative Issues', color=color2)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines3 = ax3.get_lines() + ax3_twin.get_lines()
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc='upper left')
    
    ax3.set_title('Cumulative Progress')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)

    # Add overall statistics
    stats_text = (
        f"Velocity Statistics:\n"
        f"• Total Story Points: {int(weekly_data['Story_Points'].sum())}\n"
        f"• Total Issues: {int(weekly_data['Key'].sum())}\n"
        f"• Avg Weekly Story Points: {weekly_data['Story_Points'].mean():.1f}\n"
        f"• Avg Weekly Issues: {weekly_data['Key'].mean():.1f}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.savefig(f"{output_dir}/velocity_chart.png", bbox_inches='tight', dpi=300)
    plt.close()
    log.info("Velocity trend chart saved.")

def create_burndown_chart(df, start_date, end_date, output_dir):
    df['Created'] = pd.to_datetime(df['Created'])
    df['Resolution_Date'] = pd.to_datetime(df['Resolution_Date'])
    
    # Calculate story points instead of just counting issues
    date_range = pd.date_range(start=start_date, end=end_date)
    cumulative_points = df.groupby('Created')['Story_Points'].sum().cumsum()
    completed_points = df[df['Status'] == 'Done'].groupby('Resolution_Date')['Story_Points'].sum().cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_points.index, cumulative_points.values, label='Story Points Planned', color='blue')
    plt.plot(completed_points.index, completed_points.values, label='Story Points Completed', color='green')
    
    # Add ideal burndown line
    if len(cumulative_points) > 0:
        total_points = cumulative_points.iloc[-1]
        ideal_y = np.linspace(total_points, 0, len(date_range))
        plt.plot(date_range, ideal_y, '--', label='Ideal Burndown', color='red')
    
    plt.title('Sprint Burndown (Story Points)')
    plt.xlabel('Date')
    plt.ylabel('Story Points Remaining')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/burndown_chart.png")
    plt.close()
    log.info("Burndown chart saved.")

def create_workload_chart(df, output_dir):
    plt.figure(figsize=(12, 6))
    workload = df.groupby('Assignee')['Story_Points'].sum().sort_values(ascending=True)
    
    # Create horizontal bar chart
    workload.plot(kind='barh', color='lightblue')
    plt.title('Team Workload Distribution')
    plt.xlabel('Story Points')
    plt.ylabel('Team Member')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/workload.png")
    plt.close()
    log.info("Workload distribution chart saved.")

def create_sprint_goals_chart(df, output_dir):
    """Create a comprehensive sprint progress visualization."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Calculate metrics by type
    type_metrics = df.groupby('Type').agg({
        'Key': 'count',
        'Story_Points': 'sum',
        'Status': lambda x: (x == 'Done').sum()
    }).fillna(0)
    
    # Calculate completion rates and points with proper NaN handling
    type_metrics['Completion_Rate'] = (type_metrics['Status'] / type_metrics['Key'] * 100).fillna(0).round(1)
    type_metrics['Points_Done'] = df[df['Status'] == 'Done'].groupby('Type')['Story_Points'].sum().fillna(0)
    type_metrics['Points_Total'] = df.groupby('Type')['Story_Points'].sum().fillna(0)
    type_metrics['Points_Rate'] = (type_metrics['Points_Done'] / type_metrics['Points_Total'] * 100).fillna(0).round(1)

    # 1. Progress Overview (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create grouped bar chart for completion rates
    x = np.arange(len(type_metrics.index))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, type_metrics['Completion_Rate'], width, 
                    label='Issues Completed %', color='lightblue')
    bars2 = ax1.bar(x + width/2, type_metrics['Points_Rate'], width,
                    label='Story Points Completed %', color='lightgreen')
    
    # Add target line
    ax1.axhline(y=80, color='red', linestyle='--', label='Target (80%)')
    
    # Add value labels
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f}%',
                    ha='center', va='bottom')
    
    autolabel(bars1, type_metrics['Completion_Rate'])
    autolabel(bars2, type_metrics['Points_Rate'])
    
    ax1.set_title('Completion Rates by Issue Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(type_metrics.index)
    ax1.set_ylabel('Completion Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Status Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    status_by_type = pd.crosstab(df['Type'], df['Status'])
    status_by_type.plot(kind='bar', stacked=True, ax=ax2)
    
    ax2.set_title('Issue Status Distribution by Type')
    ax2.set_xlabel('Issue Type')
    ax2.set_ylabel('Number of Issues')
    ax2.legend(title='Status', bbox_to_anchor=(1.05, 1))
    plt.setp(ax2.get_xticklabels(), rotation=45)

    # 3. Detailed Metrics Table (Bottom)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Prepare table data with safe value handling
    table_data = []
    for idx in type_metrics.index:
        row = type_metrics.loc[idx]
        points_done = row['Points_Done'] if not pd.isna(row['Points_Done']) else 0
        points_total = row['Points_Total'] if not pd.isna(row['Points_Total']) else 0
        
        table_data.append([
            idx,
            f"{int(row['Key'])}",
            f"{int(row['Status'])}",
            f"{row['Completion_Rate']:.1f}%",
            f"{int(points_total)}",
            f"{int(points_done)}",
            f"{row['Points_Rate']:.1f}%"
        ])
    
    # Add totals row with safe calculations
    total_key = type_metrics['Key'].sum()
    total_status = type_metrics['Status'].sum()
    total_completion = (total_status / total_key * 100) if total_key > 0 else 0
    total_points = type_metrics['Points_Total'].sum()
    total_points_done = type_metrics['Points_Done'].sum()
    total_points_rate = (total_points_done / total_points * 100) if total_points > 0 else 0
    
    totals = [
        'Total',
        f"{int(total_key)}",
        f"{int(total_status)}",
        f"{total_completion:.1f}%",
        f"{int(total_points)}",
        f"{int(total_points_done)}",
        f"{total_points_rate:.1f}%"
    ]
    table_data.append(totals)
    
    # Create table
    columns = ['Issue Type', 'Total\nIssues', 'Issues\nDone', 'Completion\nRate', 
               'Total Story\nPoints', 'Points\nDone', 'Points\nCompletion']
    
    table = ax3.table(cellText=table_data,
                     colLabels=columns,
                     loc='center',
                     cellLoc='center',
                     colColours=['lightgray']*len(columns))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Add sprint summary with safe calculations
    summary_text = (
        f"Sprint Summary:\n"
        f"• Overall Completion Rate: {total_completion:.1f}%\n"
        f"• Story Points Progress: {int(total_points_done)}/{int(total_points)} pts "
        f"({total_points_rate:.1f}%)\n"
        f"• Total Issues: {int(total_key)} ({int(total_status)} completed)"
    )
    
    fig.text(0.02, 0.02, summary_text, fontsize=9,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.savefig(f"{output_dir}/sprint_goals.png", bbox_inches='tight', dpi=300)
    plt.close()
    log.info("Sprint goals chart saved.")

def create_priority_chart(df, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Create stacked bar for priorities by status
    priority_status = pd.crosstab(df['Priority'], df['Status'])
    priority_status.plot(kind='bar', stacked=True)
    
    plt.title('Issue Priority Distribution by Status')
    plt.xlabel('Priority Level')
    plt.ylabel('Number of Issues')
    plt.legend(title='Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/priority_distribution.png")
    plt.close()
    log.info("Priority distribution chart saved.")

def create_developer_metrics_chart(df, output_dir):
    """Create a comprehensive developer metrics visualization."""
    # Filter for completed issues and calculate metrics
    df_done = df[df['Status'] == 'Done'].copy()
    df_done['Created'] = pd.to_datetime(df_done['Created'])
    df_done['Resolution_Date'] = pd.to_datetime(df_done['Resolution_Date'])
    df_done['Cycle_Time'] = (df_done['Resolution_Date'] - df_done['Created']).dt.days
    
    # Calculate metrics per developer
    dev_metrics = df_done.groupby('Assignee').agg({
        'Key': 'count',  # Total issues
        'Story_Points': 'sum',  # Total story points
        'Cycle_Time': ['mean', 'median', 'count']  # Cycle time stats
    }).round(2)
    
    # Flatten column names
    dev_metrics.columns = ['Issues_Completed', 'Story_Points', 'Avg_Cycle_Time', 'Median_Cycle_Time', 'Cycle_Time_Count']
    
    # Remove unassigned and inactive developers
    dev_metrics = dev_metrics[
        (dev_metrics.index != 'Unassigned') & 
        (dev_metrics['Issues_Completed'] > 0)  # Keep developers who completed any issues
    ].sort_values('Issues_Completed', ascending=False)  # Sort by completed issues
    
    if len(dev_metrics) == 0:
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No completed issues found in the selected time period',
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(f"{output_dir}/developer_metrics.png")
        plt.close()
        log.warning("No developer metrics to display")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # 1. Main metrics table (top left)
    ax_table = fig.add_subplot(gs[0, 0])
    
    # Create a table-like visualization
    table_data = []
    for dev in dev_metrics.index:
        story_points = dev_metrics.loc[dev, 'Story_Points']
        issues_completed = dev_metrics.loc[dev, 'Issues_Completed']
        avg_cycle_time = dev_metrics.loc[dev, 'Avg_Cycle_Time']
        sp_per_issue = story_points / issues_completed if issues_completed > 0 else 0
        
        table_data.append([
            dev,
            int(story_points) if not pd.isna(story_points) else 0,
            int(issues_completed),
            f"{avg_cycle_time:.1f}" if not pd.isna(avg_cycle_time) else "N/A",
            f"{sp_per_issue:.1f}" if sp_per_issue > 0 else "N/A"
        ])
    
    column_labels = ['Developer', 'Story\nPoints', 'Issues\nCompleted', 'Avg Cycle\nTime (days)', 'SP per\nIssue']
    table = ax_table.table(cellText=table_data,
                          colLabels=column_labels,
                          loc='center',
                          cellLoc='center',
                          colColours=['lightgray']*len(column_labels))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.axis('off')
    ax_table.set_title('Developer Performance Metrics', pad=20)
    
    # 2. Contribution Comparison (top right)
    ax_contrib = fig.add_subplot(gs[0, 1])
    
    # Create stacked bar for both story points and issues
    x = np.arange(len(dev_metrics.index))
    width = 0.35
    
    # Normalize the values to make them comparable
    max_sp = dev_metrics['Story_Points'].max()
    max_issues = dev_metrics['Issues_Completed'].max()
    normalized_sp = dev_metrics['Story_Points'] / max_sp if max_sp > 0 else dev_metrics['Story_Points']
    normalized_issues = dev_metrics['Issues_Completed'] / max_issues
    
    bars1 = ax_contrib.bar(x - width/2, normalized_sp, width, label='Story Points', color='lightblue')
    bars2 = ax_contrib.bar(x + width/2, normalized_issues, width, label='Issues', color='lightgreen')
    
    # Add value labels on the bars
    def autolabel(bars, original_values):
        for bar, val in zip(bars, original_values):
            height = bar.get_height()
            ax_contrib.text(bar.get_x() + bar.get_width()/2, height,
                          f'{int(val)}',
                          ha='center', va='bottom')
    
    autolabel(bars1, dev_metrics['Story_Points'])
    autolabel(bars2, dev_metrics['Issues_Completed'])
    
    ax_contrib.set_title('Contribution Comparison')
    ax_contrib.set_xticks(x)
    ax_contrib.set_xticklabels(dev_metrics.index, rotation=45)
    ax_contrib.legend()
    ax_contrib.set_ylabel('Normalized Values')
    
    # 3. Cycle Time Analysis (bottom left)
    ax_cycle = fig.add_subplot(gs[1, 0])
    cycle_times = dev_metrics['Avg_Cycle_Time']
    colors = ['green' if ct <= cycle_times.mean() else 'orange' for ct in cycle_times]
    bars = ax_cycle.bar(dev_metrics.index, cycle_times, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax_cycle.text(bar.get_x() + bar.get_width()/2, height,
                     f'{height:.1f}d',
                     ha='center', va='bottom')
    
    ax_cycle.set_title('Average Cycle Time by Developer')
    ax_cycle.set_xticklabels(dev_metrics.index, rotation=45)
    ax_cycle.axhline(y=cycle_times.mean(), color='red', linestyle='--', label='Team Average')
    ax_cycle.legend()
    
    # 4. Highlights and Analysis (bottom right)
    ax_highlights = fig.add_subplot(gs[1, 1])
    ax_highlights.axis('off')
    
    # Calculate highlights
    highlights = [
        f"Team Highlights:",
        f"• Total Story Points: {int(dev_metrics['Story_Points'].sum())}",
        f"• Total Issues: {int(dev_metrics['Issues_Completed'].sum())}",
        f"• Team Avg Cycle Time: {dev_metrics['Avg_Cycle_Time'].mean():.1f} days",
        "",
        f"Notable Achievements:",
        f"• Most Issues: {dev_metrics['Issues_Completed'].idxmax()} ({int(dev_metrics['Issues_Completed'].max())} issues)",
        f"• Most Story Points: {dev_metrics[dev_metrics['Story_Points'] > 0]['Story_Points'].idxmax() if any(dev_metrics['Story_Points'] > 0) else 'N/A'} ({int(dev_metrics['Story_Points'].max()) if any(dev_metrics['Story_Points'] > 0) else 0} pts)",
        f"• Fastest Cycle Time: {dev_metrics['Avg_Cycle_Time'].idxmin()} ({dev_metrics['Avg_Cycle_Time'].min():.1f} days)"
    ]
    
    ax_highlights.text(0, 1, '\n'.join(highlights), 
                      va='top', ha='left',
                      bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=1'))
    
    plt.savefig(f"{output_dir}/developer_metrics.png", bbox_inches='tight', dpi=300)
    plt.close()
    log.info("Developer metrics chart saved.")

def create_developer_focus_chart(df, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Calculate total tasks completed by each developer
    total_tasks = df['Key'].count()
    dev_completion = df.groupby('Assignee').agg({
        'Key': 'count'  # Total tasks completed by each developer
    })
    
    # Calculate percentage of total tasks completed by each developer
    dev_completion['Percentage'] = (dev_completion['Key'] / total_tasks * 100).round(2)
    
    # Sort by percentage and exclude 'Unassigned' if present
    if 'Unassigned' in dev_completion.index:
        dev_completion = dev_completion.drop('Unassigned')
    dev_completion_sorted = dev_completion.sort_values('Percentage', ascending=False)
    
    # Create pie chart
    dev_completion_sorted['Percentage'].plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Pastel1')
    plt.title('Percentage of Total Tasks Completed by Team Member')
    plt.ylabel('')  # Hide y-label for pie chart
    plt.tight_layout()
    plt.savefig(f"{output_dir}/developer_focus.png")
    plt.close()
    log.info("Developer focus chart saved.")

def create_developer_trends_chart(df, output_dir):
    df_done = df[df['Status'] == 'Done'].copy()
    df_done['Week'] = pd.to_datetime(df_done['Created']).dt.isocalendar().week
    df_done['Cycle_Time'] = (pd.to_datetime(df_done['Resolution_Date']) - 
                            pd.to_datetime(df_done['Created'])).dt.days
    
    plt.figure(figsize=(12, 6))
    for developer in df_done['Assignee'].unique():
        if developer != 'Unassigned':
            dev_data = df_done[df_done['Assignee'] == developer]
            weekly_avg = dev_data.groupby('Week')['Cycle_Time'].mean()
            plt.plot(weekly_avg.index, weekly_avg.values, marker='o', label=developer)
    
    plt.title('Developer Cycle Time Trends by Team Member')
    plt.xlabel('Week Number')
    plt.ylabel('Average Cycle Time (Days)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/developer_trends.png")
    plt.close()
    log.info("Developer trends chart saved.")

def create_story_points_focus_chart(df, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Calculate total story points completed by each developer
    total_story_points = df['Story_Points'].sum()
    dev_story_points = df.groupby('Assignee').agg({
        'Story_Points': 'sum'  # Total story points completed by each developer
    })
    
    # Calculate percentage of total story points completed by each developer
    dev_story_points['Percentage'] = (dev_story_points['Story_Points'] / total_story_points * 100).round(2)
    
    # Sort by percentage and exclude 'Unassigned' if present
    if 'Unassigned' in dev_story_points.index:
        dev_story_points = dev_story_points.drop('Unassigned')
    dev_story_points_sorted = dev_story_points.sort_values('Percentage', ascending=False)
    
    # Create pie chart
    dev_story_points_sorted['Percentage'].plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Pastel2')
    plt.title('Percentage of Total Story Points Completed by Team Member')
    plt.ylabel('')  # Hide y-label for pie chart
    plt.tight_layout()
    plt.savefig(f"{output_dir}/story_points_focus.png")
    plt.close()
    log.info("Story points focus chart saved.")

def main():
    args = parse_args()
    data = fetch_jira_data(args['server'], args['username'], args['token'],
                          args['project'], args['begin_date'], args['end_date'])
    
    if not data:
        log.warning("No data retrieved. Exiting.")
        return
    
    df = pd.DataFrame(data)
    df['Created'] = pd.to_datetime(df['Created'])
    
    chart_functions = {
        'velocity': lambda df, output_dir: create_velocity_trend_chart(df, output_dir),
        'cycle_time': lambda df, output_dir: create_cycle_time_chart(df, output_dir),
        'issue_distribution': lambda df, output_dir: plot_issue_distribution(df, f"{output_dir}/issue_distribution.png"),
        'burndown': lambda df, output_dir: create_burndown_chart(df, args['begin_date'], args['end_date'], output_dir),
        'throughput': lambda df, output_dir: plot_throughput(df, f"{output_dir}/throughput_chart.png"),
        'workload': lambda df, output_dir: create_workload_chart(df, output_dir),
        'sprint_goals': lambda df, output_dir: create_sprint_goals_chart(df, output_dir),
        'priority': lambda df, output_dir: create_priority_chart(df, output_dir),
        'developer_metrics': lambda df, output_dir: create_developer_metrics_chart(df, output_dir),
        'developer_focus': lambda df, output_dir: create_developer_focus_chart(df, output_dir),
        'developer_trends': lambda df, output_dir: create_developer_trends_chart(df, output_dir),
        'story_points_focus': lambda df, output_dir: create_story_points_focus_chart(df, output_dir)
    }
    
    for chart_type in args['charts']:
        if chart_type in chart_functions:
            log.info(f"Generating {chart_type} chart...")
            chart_functions[chart_type](df, args['output_dir'])
            log.info(f"Saved {chart_type} chart to {args['output_dir']}")
        else:
            log.warning(f"Unknown chart type: {chart_type}")

if __name__ == "__main__":
    main()

