# BloomAI 🌸 - Intelligent Flower Recognition System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)

BloomAI is a state-of-the-art flower recognition system that leverages deep learning to identify various flower species with high accuracy. Built using TensorFlow and Flask, this application provides an intuitive web interface for instant flower identification.

[View Demo](https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN) | [Report Bug](https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN/issues) | [Request Feature](https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN/issues)

![BloomAI Demo](screenshots/demo.gif)

## 🌟 Features

- **Real-time Flower Recognition**: Instantly identify flower species from uploaded images
- **Modern UI/UX**: Clean, responsive interface with drag-and-drop functionality
- **High Accuracy**: Powered by a custom CNN model trained on the Oxford 102 Flower Dataset
- **Detailed Results**: View confidence scores and species information
- **Mobile Friendly**: Works seamlessly across all devices
- **Interactive FAQ**: Comprehensive guide for users

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow, Keras
- **Dataset**: Oxford 102 Flower Dataset
- **Deployment**: Docker-ready

## 📋 Prerequisites

Before running the project, ensure you have:

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- Git

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN.git
cd Plant_Specis_Detection_using_CNN
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
```bash
# The model will be automatically downloaded when you first run the application
# Or you can manually place it in the models/ directory
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## 📦 Project Structure

```
Plant_Species_Detection/
├── app.py                  # Flask application
├── models/
│   └── final_model.keras   # Trained CNN model
├── static/
│   ├── css/
│   │   └── style.css      # Application styling
│   ├── js/
│   │   └── main.js        # Frontend functionality
│   └── uploads/           # Temporary image storage
├── templates/
│   └── index.html         # Main application template
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🎯 Model Architecture

Our CNN model architecture:

- Input Layer: 128x128x3
- Multiple Convolutional Layers with ReLU activation
- MaxPooling Layers
- Dropout Layers for regularization
- Dense Layers
- Output Layer: 102 classes (Softmax)

## 📊 Performance

- **Accuracy**: 94.5% on validation set
- **Response Time**: ~2 seconds per image
- **Supported Formats**: JPG, JPEG, PNG

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author

**Akash Kumar**
- GitHub: [@TechWithAkash](https://github.com/TechWithAkash)

## 🙏 Acknowledgments

- [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [TensorFlow Team](https://www.tensorflow.org/)
- [Flask Team](https://flask.palletsprojects.com/)

## 📞 Contact

- **Akash Kumar**
- Email: [vishwakarmaakshav17@gmail.com]
- Project Link: [https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN](https://github.com/TechWithAkash/Plant_Specis_Detection_using_CNN)
