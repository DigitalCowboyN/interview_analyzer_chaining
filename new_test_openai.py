import os
from openai import OpenAI

# Set your API key here or ensure it's set in your environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

async def main():  # Define an async main function
    try:
        # Test the responses.create method
        response = await client.responses.create(  # Ensure this is awaited
        model="gpt-4",
        instructions="You are a coding assistant.",
        input="What is the capital of France?"
    )
        print(response.output_text)  # Ensure this is the correct attribute to access
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())  # Ensure the main function is run
