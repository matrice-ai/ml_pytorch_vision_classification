"""Module for Session class handling project sessions."""
from matrice_sdk.rpc import RPC

class Session:
    """Class to manage sessions for a specific project."""
    def __init__(self, project_id=""):
        """Initialize a new session instance."""
        # assert project_id, "project_id is empty"
        self.rpc = RPC(project_id)
        self.project_id = project_id

    def update_project(self, project_id):
        self.project_id = project_id
        self.rpc = RPC(project_id)
    
    def close(self):
        self.rpc = None
        self.project_id = None