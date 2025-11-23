# 🏥 Skin Cancer Detection Web Application

A modern, interactive web interface for the Skin Cancer Detection deep learning model. This application allows users to upload images of skin lesions and get instant predictions about the type of skin condition.

## 🌟 Features

- **User-Friendly Interface**: Clean, intuitive design for easy navigation
- **Real-time Predictions**: Get instant classification results with confidence scores
- **Interactive Visualizations**: View prediction probabilities with interactive charts
- **Detailed Class Information**: Learn about different types of skin conditions
- **Responsive Design**: Works on desktop and mobile devices
- **Sample Images**: Try the app with example images before uploading your own

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository** (if you haven't already)
   ```bash
   git clone https://github.com/Nermine-1/skin-cancer-detection.git
   cd skin-cancer-detection
   ```

2. **Navigate to the webapp directory**
   ```bash
   cd webapp
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Open your web browser**
   The application will automatically open in your default web browser at `http://localhost:8501`

## 🖥️ How to Use

1. **Upload an Image**
   - Click on the "Browse files" button to upload a skin lesion image
   - Supported formats: JPG, JPEG, PNG

2. **View Results**
   - The application will display the uploaded image
   - Prediction results will show the most likely diagnosis
   - A confidence score indicates the model's certainty
   - Interactive charts show probability distribution across all classes

3. **Explore Class Information**
   - Use the sidebar to learn more about each skin condition
   - View detailed descriptions and characteristics of each class

4. **Try Sample Images**
   - If you don't have an image, try the example images provided

## 📱 Mobile Support

The application is fully responsive and works on mobile devices. You can access it through your mobile browser by connecting to the same network as your computer and using your computer's IP address (e.g., `http://<your-computer-ip>:8501`).

## ⚠️ Important Note

This application is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nermine**

## 🙏 Acknowledgments

- The HAM10000 dataset for providing the training data
- Streamlit for the amazing web framework
- TensorFlow/Keras for the deep learning framework
