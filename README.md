# SMAI-assignment-3
## Devanagari character predicter

A Streamlit web app that recognizes handwritten Devanagari characters using a CNN trained on the UCI dataset.
Built as part of SMAI Assignment 3 (IIIT Hyderabad).

streamlit link :
```
https://smai-assignment-3-iabwrp2zt6amdukwrz5qyj.streamlit.app/
```

##  Features

- ✏️ Draw and recognize Devanagari characters
- 🔝 Top-3 predictions with confidence scores
- 🎯 Practice mode with scoring and streaks
- 📊 Real-time feedback

##  Model

- Architecture: CNN (3 convolution layers + FC)
- Input: 32×32 grayscale images
- Classes: 46 (36 characters + 10 digits)
- Dataset: UCI Devanagari Handwritten Dataset (~92k images)
- Accuracy: ~98%

##  Installation

Clone the repo:

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## Install dependencies
pip install -r requirements.txt

## Run the app
streamlit run app.py

## 
---

##  Project Structure

```markdown.
├── app.py
├── best_model.pth
├── requirements.txt
└── README.md
```
##  Authors

- Lalithamsh K 2025202039
- Amal Agakar
- Course: SMAI (IIIT Hyderabad)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
