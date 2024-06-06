"""
This module provides a prediction interface for Cog, utilizing a pre-trained model
to generate answers based on input queries.
"""

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from dotenv import load_dotenv
from concat.concat import load_model, process_query, get_similar_chunks, generate_answer
from dotenv import load_dotenv


load_dotenv()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = load_model()
        
    def predict(
        self, 
        query:str = Input(
            description="The query for which to generate an answer",
            default="What is the answer to the question of life, the universe, and everything?"
        ), 
    
    ) -> str:
        """Predict the answer to a query using the loaded model"""
        self.model = load_model( "./wyrm.json")
        query_embedding = process_query(query)
        similar_chunks = get_similar_chunks(self.model, query_embedding)
        return generate_answer(self.model, similar_chunks, query)
