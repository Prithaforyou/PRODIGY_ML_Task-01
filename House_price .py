import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Load Data
df = pd.read_csv('train.csv')
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']].fillna(0)
y = df['SalePrice']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction Function
def predict_price(grlivarea, bedrooms, bathrooms):
    input_df = pd.DataFrame({'GrLivArea': [grlivarea],
                             'BedroomAbvGr': [bedrooms],
                             'FullBath': [bathrooms]})
    price = model.predict(input_df)[0]
    return f"Predicted House Price: ${price:.2f}"

# Gradio Web Interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Square Footage"),
        gr.Number(label="Number of Bedrooms"),
        gr.Number(label="Number of Bathrooms")
    ],
    outputs="text",
    title="House Price Predictor",
    description="Predict house price based on square footage, bedrooms, and bathrooms",
    theme="glass"
)

iface.launch()
