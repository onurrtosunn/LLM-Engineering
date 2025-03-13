import os
import gradio as gr
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv


# Initialize OpenAI client with your API key
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

client = OpenAI()

def determine_content_type(message):
    """
    Use GPT model to determine if the user wants an image or text response
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a classifier that determines if a user is requesting an image or text information. Respond with only 'IMAGE' if the user is asking for an image, visualization, or something that should be drawn. Respond with only 'TEXT' for all other requests."
                },
                {"role": "user", "content": message}
            ],
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().upper()
        return "IMAGE" if result == "IMAGE" else "TEXT"
    except Exception as e:
        print(f"Error determining content type: {str(e)}")
        return "TEXT"  # Default to text if there's an error

def generate_text_response(message):
    """
    Generate text response using GPT-4o Mini
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating text response: {str(e)}"

def generate_image_response(prompt):
    """
    Generate image response using DALL-E 3
    """
    try:
        # Enhance the prompt for better image generation
        enhanced_prompt = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating detailed prompts for DALL-E. Convert the user's request into a detailed, descriptive prompt that will create the best possible image. Include details about style, lighting, composition and mood."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        ).choices[0].message.content
        
        # Generate the image with the enhanced prompt
        response = client.images.generate(
            model="dall-e-2",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        
        # Download the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        return img, enhanced_prompt
    except Exception as e:
        return f"Error generating image: {str(e)}", None

def chatbot(message, history):
    """
    Main chatbot function that uses AI to decide between text and image response
    """
    # Eğer mesaj "Görselleştir" ise, önceki mesajı al ve görselleştir
    if message.strip().lower() in ["görselleştir", "bunu görselleştir", "resim oluştur"]:
        if history:
            last_message = history[-1][1]   # Son kullanıcının mesajını al
            content_type = "IMAGE"
        else:
            return history, "Önce bir konu hakkında konuşmalısınız."
    else:
        last_message = message
        content_type = determine_content_type(message)

    if content_type == "IMAGE":
        
        response_image, enhanced_prompt = generate_image_response(last_message)
        print("Generated Prompt:", enhanced_prompt)

        if isinstance(response_image, str):  # Hata oluştuysa
            history.append((message, response_image))
            return history, None
        else:
            response_text = f"Here's the image you requested.\n\nPrompt used: {enhanced_prompt}"
            history.append((message, response_text))
            return history, response_image
    else:
        response_text = generate_text_response(message)
        history.append((message, response_text))
        return history, None

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant with Intelligent Content Type Selection")
    gr.Markdown("Ask anything! The AI will automatically determine whether to respond with text or an image.")
    
    chatbot_interface = gr.Chatbot()
    image_output = gr.Image(label="Generated Image")
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.Button("Clear")
    
    msg.submit(chatbot, [msg, chatbot_interface], [chatbot_interface, image_output])
    clear.click(lambda: [], None, chatbot_interface, queue=False)
    clear.click(lambda: None, None, image_output, queue=False)
    
    # Examples with inputs parameter
    example_inputs = [
        "Tell me about artificial intelligence",
        "A sunset over the ocean",
        "What is machine learning?",
        "A cat playing with a ball of yarn",
        "Explain how neural networks work",
        "A futuristic city with flying cars"
    ]
    
    gr.Examples(
        examples=example_inputs,
        inputs=msg
    )
    
    gr.Markdown("## How to use")
    gr.Markdown("""
    - Simply type your question or request
    - The AI will automatically determine whether to respond with text or an image
    - For images, the AI will enhance your prompt to create the best possible image
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)