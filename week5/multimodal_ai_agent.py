import os
import json
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from PIL import Image

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI()

SYSTEM_MESSAGE = "You are a helpful assistant for an Airline called FlightAI. Give short, courteous answers, no more than 1 sentence. Always be accurate. If you don't know the answer, say so."
TICKET_PRICES = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


class ChatBot:
    """
    Handles GPT-based conversation, including ticket price retrieval and image generation.
    """

    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def get_ticket_price(self, city):
        """
        Returns the round-trip ticket price for a given city.
        """
        city = city.lower()
        return TICKET_PRICES.get(city, "Unknown")

    def handle_tool_call(self, message):
        """
        Processes tool calls for ticket pricing and image generation.
        """
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "get_ticket_price":
            city = arguments.get("destination_city")
            price = self.get_ticket_price(city)
            response = {"role": "tool", "content": json.dumps({"destination_city": city, "price": price}),
                        "tool_call_id": tool_call.id}
            return response, None

        elif tool_call.function.name == "generate_image":
            prompt = arguments.get("prompt")
            if not prompt:
                return {"role": "tool", "content": "No prompt provided for image generation."}, None
            
            image = ImageGenerator().generate_image(prompt)
            description = self.describe_image(prompt)

            response = {"role": "tool", "content": description, "tool_call_id": tool_call.id}
            return response, image

    def describe_image(self, prompt):
        """
        Generates a short description of the AI-generated image using GPT.
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI that shortly describes images in Turkish."},
                {"role": "user", "content": f"Describe shortly an AI-generated image of: {prompt} in Turkish"}
            ]
        )
        return response.choices[0].message.content

    def chat(self, history):
        """
        Handles user chat interactions, including calling tools for image generation or ticket prices.
        """
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history
        response = openai.chat.completions.create(model=self.model, messages=messages, tools=GradioUI.tools)

        image = None
        audio_path = None

        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_response, image = self.handle_tool_call(message)
            messages.append(message)
            messages.append(tool_response)
            reply = tool_response["content"]
        else:
            reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": reply})
        audio_path = AudioGenerator().generate_audio(reply)

        return history, image, audio_path

class ImageGenerator:
    """
    Handles AI-generated images using the DALL·E model.
    """
    def generate_image(self, prompt):
        """
        Generates an image based on the provided prompt using DALL·E.
        """
        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))

class AudioGenerator:
    """
    Converts text responses into speech audio.
    """
    def generate_audio(self, text):
        """
        Generates an AI voice response from text.
        """
        response = openai.audio.speech.create(
            model="tts-1",
            voice="ash",
            input=text
        )
        audio_path = "/tmp/response_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path

class GradioUI:
    """
    Manages the Gradio interface for the AI chatbot, including text, images, and audio output.
    """
    
    tools = [
        {"type": "function", "function": {
            "name": "get_ticket_price",
            "description": "Get the price of a return ticket to the destination city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {"type": "string", "description": "The city that the customer wants to travel to"}
                },
                "required": ["destination_city"],
                "additionalProperties": False
            }
        }},
        {"type": "function", "function": {
            "name": "generate_image",
            "description": "Generate an image based on user input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt describing the image to be generated"}
                },
                "required": ["prompt"],
                "additionalProperties": False
            }
        }}
    ]

    def __init__(self):
        self.chatbot = ChatBot()
        self.build_ui()

    def build_ui(self):
        """
        Creates and launches the Gradio user interface.
        """
        with gr.Blocks() as ui:
            with gr.Row():
                chatbot = gr.Chatbot(height=500, type="messages")
                image_output = gr.Image(height=500)
            with gr.Row():
                entry = gr.Textbox(label="Chat with our AI Assistant:")
            with gr.Row():
                clear = gr.Button("Clear")
            with gr.Row():
                audio_output = gr.Audio(label="AI Voice Response", autoplay=True)

            def do_entry(message, history):
                history.append({"role": "user", "content": message})
                return "", history

            entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
                self.chatbot.chat, inputs=chatbot, outputs=[chatbot, image_output, audio_output]
            )
            clear.click(lambda: None, inputs=None, outputs=[chatbot, audio_output], queue=False)

        ui.launch(share=True)

if __name__ == "__main__":
    GradioUI()
