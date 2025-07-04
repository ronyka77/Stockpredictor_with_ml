# Base Development Rules

## Logging Standards

### Rule: Use Centralized Logging System
All modules MUST use the centralized logging system from `src/logger.py`.

### Rule: Use Configuration Files for Parameters
All parameters MUST come from config.py files or environment variables (.env). Direct hardcoding of configuration values is prohibited.

### Rule: No Argument Parsing
Argument parsing (argparse, click, etc.) MUST NOT be used in any file under any circumstances. All runtime configuration should be handled through config files or environment variables.

