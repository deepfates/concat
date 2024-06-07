# 🐈‍⬛ concat

This is a simple retrieval chat api. It takes a Bookwyrm object and a text query, does similarity search to find relevant docs, and outputs a synthetic answer with citations.

To use, create a virtual environment and install requirements with pip.

Here's how you can use concat with the [bookwyrm](https://github.com/deepfates/bookwyrm) model you created:

```python
from concat.concat import load_model, get_full_answer

# Load the Bookwyrm model
model = load_model("wyrm.json")

# Ask a query
query = "What is the meaning of life?"

# Get the answer
answer = "".join(get_full_answer(model, query))
print(answer)
```

This will:

1. Load the `bookwyrm` model from the `wyrm.json` file you created.
2. Process the query "What is the capital of France?" to get its embedding.
3. Find the most similar chunks in the model to the query embedding.
4. Generate a prompt with the query and relevant chunks.
5. Use the `meta/meta-llama-3-8b-instruct` model to generate an answer based on the prompt.
6. Print out the full answer with citations.

To use `concat` with Replicate, update the `predict.py` file:

```python
from concat.concat import load_model, get_full_answer

class Predictor(BasePredictor):
    def setup(self):
        self.model = load_model("wyrm.json")

    def predict(self, query: str) -> str:
        return "".join(get_full_answer(self.model, query))
```

Then build and push to Replicate as shown in the README.
