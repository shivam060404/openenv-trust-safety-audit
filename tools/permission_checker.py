"""
A simulated clearance/permission checker tool.
"""

def check_clearance(user_id: str, database: dict) -> str:
    """
    Checks if a user has clearance.
    """
    if user_id in database:
        return f"CLEARANCE for user {user_id}: {database[user_id]}"
    return f"UNKNOWN USER: {user_id}. Assume no special clearance."
