# Email Spam Classifier (Machine Learning Project)

A supervised machine learning project that classifies emails as **Spam** or **Not Spam (Ham)** using natural language processing (TF-IDF) and ensemble models.  
This app allows users to paste an email or upload a PDF to check whether it’s spam.

## Features

- Trained on real-world spam and ham email data  
- Uses TF-IDF vectorization for text-to-number conversion  
- Random Forest classifier for reliable performance  
- Streamlit app for easy, interactive predictions  
- Option to test emails by pasting text or uploading PDF files  

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Fahad-Rehman/email-spam-classifier.git
cd email-spam-classifier
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare the dataset
Place your raw dataset in:
```bash
data/spam/
data/ham/
```
Each folder should contain .txt email files.

You can download a similar dataset here:
https://www2.aueb.gr/users/ion/data/enron-spam/
### 4. Train the model
```bash
python train_text_model.py
```
This will generate:
```bash
model/model.pkl
model/vectorizer.pkl
```
### 5. Run the streamlit app
```bash
streamlit run app.py
```
Then open the local URL shown and test your email samples or PDFs

## Model Details

- Vectorizer: TF-IDF (max_features=5000)
- Model: RandomForestClassifier (n_estimators=300, random_state=42)
- Metric: Accuracy, Precision, Recall, F1-Score
- You can easily switch to other models (e.g. Logistic Regression, XGBoost) in train_text_model.py.

## Future Improvements

- Add deep learning (BERT) for semantic understanding
- Include email header metadata (sender info, links)
- Deploy on Streamlit Cloud or Hugging Face Spaces

## License

This project is open-source under the MIT License.

## Author

Fahad Khatri
Pakistan