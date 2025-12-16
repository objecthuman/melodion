# Project Rules

## Code Style

- **No unnecessary docstrings or comments**: Do not write docstrings or comments where the code is self-explanatory. Function names, parameter names, and type hints should make the intent clear.
- Keep code concise and readable through good naming conventions rather than excessive documentation.
- **Proper type hints**: Always use proper TypedDict for structured data returns instead of generic `dict`. Create TypedDict classes where necessary for better type safety.

## Logging

- **Always use structured logging**: Use the `Logger` dependency in FastAPI endpoints to get the request-bound logger.
- **Log with context**: Pass relevant context as keyword arguments (e.g., `logger.info("message", file_count=10, batch_size=32)`).
- **Log levels**:
  - `logger.info()`: Successful operations, important state changes
  - `logger.warning()`: Invalid input, recoverable errors
  - `logger.error()`: Exceptions and failures - always use `exc_info=True` to include stack traces
- **Never use print()**: Always use the logger instead of print statements.
