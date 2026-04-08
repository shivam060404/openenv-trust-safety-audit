import sys
import os

# Add parent directory to path to find the actual server.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

def main() -> None:
    """Entry point for openenv validation checking."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
