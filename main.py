import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

from inference import process_image_extraction

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True,revision='refs/pr/6')
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

# model=model.to(device)

# Define the entity types from the problem statement
ENTITY_TYPES = [
    "width",
    "depth",
    "height",
    "item_weight",
    "maximum_weight_recommendation",
    "voltage",
    "wattage",
    "item_volume"
]

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

# def run_example(task_prompt, text_input, image):
#     prompt = task_prompt + text_input

#     # Ensure the image is in RGB mode
#     if image.mode != "RGB":
#         image = image.convert("RGB")

#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3
#     )
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#     parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
#     return parsed_answer


# def get_base_prompt(entity_name):
#     entity_name_=entity_name.lower().replace('_', ' ')
#     base_prompt=f"""
#       Extract {entity_name_} of product in the given image.
#       Use entity unit as one of allowed units:
#       {entity_unit_map[entity_name]}
#       """
#     return base_prompt

# def predict_entity(input_image, entity_name):
#     # """
#     # Placeholder function for prediction.
#     # In a real implementation, this would contain your ML model logic.
#     # """
#     # if entity_name == "item_weight":
#     #     return "500 gram"
#     # elif entity_name in ["width", "height", "depth"]:
#     #     return "10 centimetre"
#     # elif entity_name == "voltage":
#     #     return "220 volt"
#     # elif entity_name == "wattage":
#     #     return "1000 watt"
#     # elif entity_name == "item_volume":
#     #     return "1 litre"
#     # else:
#     #     return ""
#     prompt=get_base_prompt(entity_name)
#     prected_answer=run_example("DocVQA", 'what is weight of product in given image?', input_image)
#     return prected_answer['DocVQA']

def process_image(image, entity_name):
    """
    Main function to process the image and return predictions
    """
    if image is None:
        return "Please upload an image"
    
    if not entity_name or entity_name=="None":
        return "Please select an entity type"
    
    try:
        # Get prediction
        entity_name_=entity_name.replace('_', ' ')
        # prediction = predict_entity(image, entity_name)
        result=process_image_extraction(image, entity_name)
        if result["status"] == "success":
            return f"Predicted {entity_name_}: {result['extracted_value']}"
        else:
            return f"\nError: {result['message']}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# Product Entity Extraction from Images")
    gr.Markdown("Upload an image and select the entity type to extract information.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            image_input = gr.Image(
                label="Upload Product Image",
                type="pil",
                height=300
            )
            entity_dropdown = gr.Dropdown(
                choices=["None"] + ENTITY_TYPES,
                label="Select Entity Type",
                value="None",
                allow_custom_value=False
            )
            submit_btn = gr.Button("Extract Entity", variant="primary", interactive=False)
            clear_btn = gr.Button("Clear")

        with gr.Column():
            # Output components
            output_text = gr.Textbox(
                label="Prediction Result",
                lines=2,
                placeholder="Predicted Item Weight: 100.0 gram",
                interactive=False
            )
    
    # Update submit button state based on inputs
    def update_button_state(image, entity_name):
        """
        Update the submit button state based on the image and entity name inputs.
        Returns whether the button should be interactive or not.
        """
        return gr.Button(value="Extract Entity", variant="primary", 
                        interactive=(image is not None and entity_name not in [None, "", "None"]))
    
    # Add change event handlers
    image_input.change(
        fn=update_button_state,
        inputs=[image_input, entity_dropdown],
        outputs=submit_btn
    )
    
    entity_dropdown.change(
        fn=update_button_state,
        inputs=[image_input, entity_dropdown],
        outputs=submit_btn
    )
    
    # Handle submit button click
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, entity_dropdown],
        outputs=output_text
    )
    
    # Handle clear button click
    def clear_fields():
        return [
            None,  # image
            "None",  # entity dropdown
            "",  # output text
            gr.Button(value="Extract Entity", variant="primary", interactive=False)  # submit button
        ]
    
    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[image_input, entity_dropdown, output_text, submit_btn]
    )
    
    gr.Markdown("""
    ### Instructions:
    1. Upload a product image using the upload button
    2. Select the entity type you want to extract from the dropdown menu
    3. Click 'Extract Entity' to get the prediction
    4. Use 'Clear' to reset all fields
    
    Note: The Extract Entity button will be enabled only after both an image is uploaded and an entity type is selected.
    """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
