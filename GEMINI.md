=====================================================
プロジェクトのルール
=====================================================
プロジェクトで作業する際は、エージェントの全ての操作（ファイルの読み書き、コマンド実行など）を記録するために、必ず `mcp_logger.py` を `run_shell_command` を通じて使用してください。

ログの記録形式：
python mcp_logger.py --level [LEVEL] --operation '[OPERATION]' --message '[MESSAGE]' --context '{"key": "value"}'

例：
python mcp_logger.py --level INFO --operation 'file_read' --message 'Read main.py' --context '{"file_path": "/path/to/main.py"}'
=====================================================
