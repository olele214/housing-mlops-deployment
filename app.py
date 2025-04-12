import pandas as pd
import numpy as np
import gradio as gr
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_price(area, bedrooms, bathrooms, stories, parking):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]],
                              columns=["area", "bedrooms", "bathrooms", "stories", "parking"])
    prediction = model.predict(input_data)
    return f"Estimated Price: {prediction[0]:,.2f}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (in sqft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Stories"),
        gr.Number(label="Parking Spaces")
    ],
    outputs="text",
    title="Housing Price Predictor",
    description="Enter housing features to predict the estimated price."
)

if __name__ == "__main__":
    interface.launch()
