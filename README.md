# Sharma - Your Friendly Chatbot Assistant

Sharma is a web-based chatbot built with Python and Flask that provides a friendly conversational interface for users.

## Features

- Clean, modern UI with responsive design
- Real-time chat interaction
- Conversation history tracking
- Rule-based responses (expandable to more advanced NLP)
- Easy to customize and extend

## Installation

1. Clone this repository or download the files
2. Create a virtual environment (recommended)
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

3. Start chatting with Sharma!

## Customization

You can customize Sharma's responses by modifying the `generate_response` function in `app.py`. 

For more advanced functionality, consider integrating:
- Natural Language Processing libraries (NLTK, spaCy)
- Machine Learning frameworks (TensorFlow, PyTorch)
- External APIs for weather, news, etc.

## Project Structure

```
python-chatbot/
├── app.py                  # Main application file
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css       # Stylesheet
│   └── js/
│       └── script.js       # Client-side JavaScript
└── templates/
    └── index.html          # Main HTML template
```

## License

MIT

## Author

Created with ❤️ using Python and Flask
