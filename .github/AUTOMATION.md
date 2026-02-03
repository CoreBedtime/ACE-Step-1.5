# Repository Automation

## Scheduled Repository Public Status

This repository has a GitHub Actions workflow configured to automatically make the repository public at a scheduled time.

### Schedule Details
- **Date**: February 3, 2026
- **Time**: 9:00 AM Pacific Standard Time (PST) / 17:00 UTC
- **Workflow**: `.github/workflows/make-repo-public.yml`

### How It Works
The workflow uses a cron schedule to trigger at the specified time and calls the GitHub API to change the repository's visibility from private to public.

### Manual Trigger
If needed, the workflow can also be triggered manually from the GitHub Actions tab using the "workflow_dispatch" event.

### Permissions
The workflow requires `contents: write` permission to modify the repository settings via the GitHub API.
