"""
OpenEnv Client for Trust & Safety Audit Environment.
"""
from openenv.core import GenericEnvClient
from models import AuditAction, AuditObservation

class TrustSafetyAuditEnvClient(GenericEnvClient[AuditAction, AuditObservation, dict]):
    """
    Client for connecting to a remote or local Trust & Safety Audit Environment.
    Provides typed methods for reset, step, and state.
    """
    pass
