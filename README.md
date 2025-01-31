# Jira Sprint Analytics

A Python-based tool for generating comprehensive sprint analytics and visualizations from Jira data.

## Features

- **Velocity Tracking**: Weekly velocity trends with story points and issue counts
- **Cycle Time Analysis**: Distribution of completion times by issue type
- **Sprint Goals**: Progress tracking against sprint targets
- **Developer Metrics**: Individual and team performance analytics
- **Workload Distribution**: Team workload analysis
- **Burndown Charts**: Sprint progress tracking
- **Issue Distribution**: Analysis by type and status

## Charts Generated

1. Velocity Chart (`velocity_chart.png`)
   - Weekly velocity trends
   - Story points vs issues comparison
   - Cumulative progress

2. Cycle Time Chart (`cycle_time_chart.png`)
   - Cycle time distribution by issue type
   - Statistical analysis
   - Individual issue markers

3. Developer Metrics (`developer_metrics.png`)
   - Individual performance metrics
   - Contribution comparison
   - Cycle time analysis
   - Team highlights

4. Sprint Goals (`sprint_goals.png`)
   - Completion rates by issue type
   - Status distribution
   - Detailed metrics table

5. Additional Charts
   - Burndown Chart (`burndown_chart.png`)
   - Issue Distribution (`issue_distribution.png`)
   - Workload Distribution (`workload.png`)
   - Priority Distribution (`priority_distribution.png`)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jira-sprint-analytics.git
   cd jira-sprint-analytics
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Jira credentials:
   ```
   JIRA_URL=https://your-domain.atlassian.net
   JIRA_USERNAME=your-email@domain.com
   JIRA_API_TOKEN=your-api-token
   ```

## Usage

Run the script with default settings:
```bash
python sprint_progress_chart.py
```

Or specify custom parameters:
```bash
python sprint_progress_chart.py --begin_date 2024-01-01 --end_date 2024-01-31 --project SCRUM
```

### Command Line Arguments

- `--server`: Jira server URL (default: from .env)
- `--username`: Jira username (default: from .env)
- `--token`: Jira API token (default: from .env)
- `--project`: Jira project key (default: SCRUM)
- `--begin_date`: Start date (YYYY-MM-DD)
- `--end_date`: End date (YYYY-MM-DD)
- `--charts`: List of charts to generate
- `--output-dir`: Directory for output charts

## Requirements

- Python 3.8+
- pandas
- matplotlib
- seaborn
- jira
- python-dotenv

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 