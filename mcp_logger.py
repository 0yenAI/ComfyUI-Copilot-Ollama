import argparse
import json
from vibe_mcp.vibelogger import create_file_logger

def main():
    """
    Vibe Logger Master Control Program (MCP).
    This script provides a command-line interface to generate structured,
    AI-friendly logs for operations performed by the Gemini CLI agent.
    """
    # The project name for agent logs, to keep them separate from application logs.
    project_name = "vibe_mcp_agent"

    parser = argparse.ArgumentParser(
        description="Vibe Logger MCP: A command-line interface for structured logging."
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Log level (e.g., INFO, ERROR)."
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        help="A short, machine-readable string identifying the operation (e.g., 'file_write', 'shell_command')."
    )
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="A human-readable message describing the event."
    )
    parser.add_argument(
        "--context",
        type=str,
        default="{}",
        help="A JSON string containing additional key-value context for the log entry."
    )
    parser.add_argument(
        "--human_note",
        type=str,
        help="A natural language note for human observers."
    )
    parser.add_argument(
        "--ai_todo",
        type=str,
        help="A task or instruction for an AI to perform based on this log."
    )

    args = parser.parse_args()

    # Create a logger instance. Logs will be saved in ./logs/<project_name>/
    logger = create_file_logger(project_name)

    # Parse the context string from JSON to a Python dictionary.
    try:
        context_dict = json.loads(args.context)
    except json.JSONDecodeError:
        # If parsing fails, log the error and the invalid string.
        logger.error(
            operation="mcp_parsing_error",
            message="Failed to parse context JSON from command line.",
            context={"invalid_context_string": args.context}
        )
        print(f"Error: Invalid JSON format for --context. See log for details.")
        return

    # Get the appropriate logging method based on the --level argument.
    # Defaults to logger.info if an unexpected level is somehow passed.
    log_method = getattr(logger, args.level.lower(), logger.info)

    # Record the log entry with all the provided details.
    log_method(
        operation=args.operation,
        message=args.message,
        context=context_dict,
        human_note=args.human_note,
        ai_todo=args.ai_todo
    )

    print(f"Log entry created successfully in './logs/{project_name}/'")

if __name__ == "__main__":
    main()
