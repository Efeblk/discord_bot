import requests
import discord
import replicate
import os
from io import BytesIO
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

# Get the tokens from environment variables
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Initialize Replicate client using the token
client_replicate = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Function to call the Replicate API and refine the prompt using meta-llama model
async def get_model_response(prompt):
    input_data = {
        "top_p": 0.9,
        "prompt": prompt,
        "min_tokens": 0,
        "temperature": 0.6,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant, short answer<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 1.15
    }

    # Initialize an empty string to hold the final response
    response = ""

    try:
        for event in client_replicate.stream("meta/meta-llama-3-70b-instruct", input=input_data):
            if event.data:  # Ensure there is data in the event
                response += event.data  # Concatenate the text from the event
    except Exception as e:
        print(f"Error while processing the response: {e}")
        response = "There was an error generating a response."

    return response

async def generate_image(refined_prompt):
    input_data = {
        "prompt": refined_prompt
    }

    try:
        # Run the image generation model
        output = client_replicate.run(
            "black-forest-labs/flux-schnell",
            input=input_data
        )

        # Check if output contains a FileOutput object
        if isinstance(output, list) and output and isinstance(output[0], replicate.helpers.FileOutput):
            image_data_url = output[0].url

            # Check if the URL is a base64-encoded data URL
            if image_data_url.startswith('data:'):
                # Extract the base64 encoded part of the URL
                base64_data = image_data_url.split(',', 1)[1]

                # Decode the base64 string into binary data
                image_data = base64.b64decode(base64_data)

                # Create a BytesIO object to hold the binary image data
                return BytesIO(image_data)
            else:
                # If it's not a base64 URL, handle it as a regular URL (downloadable image)
                response = requests.get(image_data_url)
                if response.status_code == 200:
                    # Create a BytesIO object to hold the image in memory
                    return BytesIO(response.content)
                else:
                    return None
        else:
            return None
    except Exception as e:
        print(f"Error while generating the image: {e}")
        return None


# Function to call the image-question answering model (for analyzing uploaded images)
async def get_image_response(image_url, prompt):
    input_data = {
        "image": image_url,
        "prompt": prompt
    }

    response = ""

    try:
        for event in client_replicate.stream(
            "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
            input=input_data
        ):
            if event.data:
                response += event.data
    except Exception as e:
        print(f"Error while processing the image: {e}")
        response = "There was an error processing the image."

    return response

# Event handler when the bot is ready
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

# Event handler for incoming messages
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if the message contains an image attachment for analysis
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']):
                await message.channel.send("I see you uploaded an image! Let me analyze it...")

                # Provide a prompt for the visual question answering model
                prompt = "explain the image in detail, short answer"

                # Get the image response
                response = await get_image_response(attachment.url, prompt)

                # Send the response back to the Discord channel
                await message.channel.send(response)
                return

    # Handling the "!generate" command for image generation
    if message.content.startswith('!generate'):
        user_prompt = message.content[10:].strip()  # Remove the '!generate ' part
        if not user_prompt.strip():  # Check for empty prompt
            await message.channel.send("Please provide a valid prompt for image generation.")
            return
        
        user_prompt = f'refine this prompt without changing the core of the prompt for an image generation AI "{user_prompt}"\n just write the refined prompt, keep it concise and do not add extra details.'

        await message.channel.send("Refining your prompt for the image... 🔄")

        # Step 1: Get a refined prompt using the text model
        refined_prompt = await get_model_response(user_prompt)

        if not refined_prompt or "There was an error" in refined_prompt:
            await message.channel.send("Sorry, something went wrong while refining your prompt.")
            return

        await message.channel.send(f"Refined Prompt: {refined_prompt}")

        # Step 2: Generate the image using the refined prompt
        await message.channel.send("Generating your image... 🖼️")
        image_data = await generate_image(refined_prompt)

        # Step 3: Send the generated image file to the Discord channel if available
        try:
            if image_data:
                await message.channel.send(file=discord.File(image_data, 'generated_image.png'))
        except Exception as e:
            print(f"{e} Error while sending the generated image ")
            await message.channel.send("Sorry, something went wrong while sending the generated image.")

    # Handling the "!ask" command for general question answering
    if message.content.startswith('!ask'):
        prompt = message.content[5:]  # Extract the prompt after the command
        if prompt:
            await message.channel.send("Let me think... 🤔")
            response = await get_model_response(prompt)
            await message.channel.send(response)
        else:
            await message.channel.send("Please provide a question after the `!ask` command.")

# Run the bot
client.run(DISCORD_TOKEN)
