# filename: test_openai_responses.py
import os
import asyncio
from openai import OpenAI

def get_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    return key

async def main():
    try:
        # Initialize an OpenAI client
        client = OpenAI(api_key=get_api_key())

        # Call the brand-new `responses.create` method
        response = await client.responses.create(
            model="gpt-4",
            instructions="You are a coding assistant.",
            input="What is the capital of France?"
        )

        # Assuming the returned response has an `output_text` attribute
        print("Model answer:", response.output_text)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the async main() when invoked directly
if __name__ == "__main__":
    asyncio.run(main())
