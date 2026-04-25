"""
A simulated policy lookup tool for the Trust & Safety Agent.
Simulates a dynamic policy database that can change to test adaptation.
"""

def lookup_policy(query: str, current_policies: dict) -> str:
    """
    Looks up a policy given a keyword or query against the current active policies.
    """
    query = query.lower()
    for key, value in current_policies.items():
        if key in query:
            return f"POLICY FOUND [{key.upper()}]: {value}"
    
    return "NO SPECIFIC POLICY FOUND for that query. Follow general safety guidelines."
