"""
This module provides a prediction interface for Cog, utilizing a pre-trained model
to generate answers based on input queries.
"""
from cog import BasePredictor, Input, ConcatenateIterator  # Use ConcatenateIterator
from dotenv import load_dotenv

from concat.concat import load_model, get_full_answer
load_dotenv()

class Predictor(BasePredictor):
    def __init__(self):
        self.model = None

    def setup(self, _ = None) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = load_model()

    def predict(
            self, 
            query: str = 
            Input(
                description="The query", 
                default="What documentation have you read?"
                )
            ) -> ConcatenateIterator[str]:
        """Predict the answer to a query using the loaded model."""
        for part in get_full_answer(self.model, query):
            yield part
