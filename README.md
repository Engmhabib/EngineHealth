
# Engine Health Analysis Dashboard

## Description
In this project, we aim to uncover the intricate relationships between various engine performance metrics and their impact on engine health. By examining metrics such as engine RPM, lubricating oil pressure, and coolant temperature, among others, we seek to answer pertinent questions regarding engine health
## Features
- Predict engine health in real-time.
- Input features include engine RPM, lubrication oil pressure, fuel pressure, coolant pressure, oil temperature, and coolant temperature.
- Outputs a binary prediction indicating normal or abnormal engine conditions.

## Installation

### Prerequisites
- Python 3.8 or later
- pip
- Virtual environment (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/engine-health-predictor.git
   cd engine-health-predictor
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the web application locally:

1. Start the server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`.

3. Enter the engine parameters into the input fields and submit them to receive the health status prediction.

## How It Works
The application uses a pretrained neural network model that takes six inputs corresponding to various engine readings. These readings are processed by the model to predict whether the engine condition is normal or abnormal.

## Contributing
Contributions to the Engine Health Predictor are welcome!

Here are some ways you can contribute:
- by reporting bugs
- by suggesting enhancements
- by writing or improving documentation
- by writing specifications
- by writing code (no patch is too small: fix typos, add comments, clean up inconsistent whitespace)
- by reviewing patches

## License
This project is done for my Final submission of SP24: VISUALIZATION DES, ANLS & EVAL: 21392

---

