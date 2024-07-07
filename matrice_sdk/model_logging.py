"""Module for handling logging of model training epochs in projects."""


class ModelLogging:
    """Class for managing model logging."""

    def __init__(self, session, model_id=None):
        """Initialize ModelLogging instance."""
        self.model_id = model_id
        self.rpc = session.rpc

    def get_model_training_logs(self):
        """Get training logs for the specified model."""
        path = f"/v1/model_logging/model/{self.model_id}/train_epoch_logs"
        resp = self.rpc.get(path=path)
        if resp.get("success"):
            error = None
            message = "Got all the models inside the project"
        else:
            error = resp.get("message")
            message = "Could not fetch model logs inside the project"

        return resp, error, message
