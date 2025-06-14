import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os # Keep this for local file operations like model saving
import joblib
import json # To parse the service account key

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Correct Google Authentication import for service accounts
from google.oauth2.service_account import Credentials # For gspread
from googleapiclient.discovery import build # For Google Drive API

# --- Configuration ---
MODEL_FOLDER_ID = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac" # Your Google Drive Folder ID
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'} # Cards already out of play for Player A's perspective
PREDICTION_ROUNDS_CONSIDERED = 10
STREAK_THRESHOLD = 3
OVER_UNDER_BIAS_THRESHOLD = 0.6

PATTERNS_TO_WATCH = {
    'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUU_O': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OUOU': ['Over 21', 'Under 21', 'Over 21', 'Under 21'],
    'UOUO': ['Under 21', 'Over 21', 'Under 21', 'Over 21'],
    'OO': ['Over 21', 'Over 21'],
    'UU': ['Under 21', 'Under 21'],
    'O_U': ['Over 21', 'Under 21'],
    'U_O': ['Under 21', 'Over 21'],
    'E_O': ['Exactly 21', 'Over 21'],
    'E_U': ['Exactly 21', 'Under 21'],
    'O_E': ['Over 21', 'Exactly 21'],
    'U_E': ['Under 21', 'Exactly 21']
}

# --- Card Definitions ---
card_values = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11 # 'A' assumed to be 11 initially
}
suits = ['♠', '♣', '♦', '♥']
all_cards_list = [f"{rank}{suit}" for rank in card_values for suit in suits]

# --- Google Sheets Authentication and Data Handling ---
@st.cache_resource(ttl=3600) # Cache connection for 1 hour
def get_gspread_client():
    try:
        # Load service account info from Streamlit secrets
        gcp_service_account_info = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n'), # Handle multiline private key
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
        }
        
        creds = Credentials.from_service_account_info(gcp_service_account_info)
        client = gspread.authorize(creds)
        st.success("Successfully connected to Google Sheets!")
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        st.info("Please ensure your `.streamlit/secrets.toml` is correctly configured and your service account has access to the Google Sheet.")
        return None

@st.cache_data(ttl=60) # Cache data for 1 minute
def load_all_historical_rounds_from_sheet():
    client = get_gspread_client()
    if not client:
        return pd.DataFrame() # Return empty if client connection failed
    
    try:
        spreadsheet = client.open("Casino Card Game Log") # Name of your Google Sheet
        worksheet = spreadsheet.worksheet("Sheet1") # Assuming data is in Sheet1
        
        data = worksheet.get_all_values()
        if not data:
            st.warning("Google Sheet 'Casino Card Game Log' is empty.")
            return pd.DataFrame()

        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        
        # Convert relevant columns to numeric, coercing errors
        numeric_cols = ['Card 1 Value', 'Card 2 Value', 'Card 3 Value', 'Dealer Card Value', 'Player Sum']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential numeric columns became NaN due to coercion
        df.dropna(subset=numeric_cols, inplace=True)

        st.success(f"Loaded {len(df)} historical rounds from Google Sheet.")
        return df
    except SpreadsheetNotFound:
        st.error("Google Sheet 'Casino Card Game Log' not found. Please ensure the name is correct and shared with the service account.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from Google Sheet: {e}")
        return pd.DataFrame()

def log_round_to_sheet(round_data):
    client = get_gspread_client()
    if not client:
        st.error("Cannot log round: Google Sheets client not available.")
        return False
    
    try:
        spreadsheet = client.open("Casino Card Game Log")
        worksheet = spreadsheet.worksheet("Sheet1")
        
        # Prepare row to append
        row_values = [
            round_data['Date'],
            round_data['Card 1'],
            round_data['Card 1 Value'],
            round_data['Card 2'],
            round_data['Card 2 Value'],
            round_data['Card 3'],
            round_data['Card 3 Value'],
            round_data['Dealer Card'],
            round_data['Dealer Card Value'],
            round_data['Player Sum'],
            round_data['Round Outcome']
        ]
        
        worksheet.append_row(row_values)
        st.success("Round logged successfully to Google Sheet!")
        # Clear cache for historical data to force reload
        load_all_historical_rounds_from_sheet.clear() 
        return True
    except Exception as e:
        st.error(f"Error logging round to Google Sheet: {e}")
        return False

# --- Google Drive API (for models) ---
@st.cache_resource(ttl=3600) # Cache Drive service for 1 hour
def get_drive_service():
    try:
        # Load service account info from Streamlit secrets
        gcp_service_account_info = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n'), # Handle multiline private key
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
        }

        creds = Credentials.from_service_account_info(gcp_service_account_info)
        drive_service = build('drive', 'v3', credentials=creds)
        st.success("Successfully connected to Google Drive!")
        return drive_service
    except Exception as e:
        st.error(f"Error connecting to Google Drive: {e}")
        st.info("Please ensure your `.streamlit/secrets.toml` is correctly configured and your service account has access to the Google Drive folder.")
        return None

def upload_model_to_drive(drive_service, folder_id, file_path, mime_type):
    try:
        file_name = os.path.basename(file_path)
        
        # Check if file exists in Drive already
        file_list = drive_service.files().list(
            q=f"'{folder_id}' in parents and name='{file_name}'",
            fields="files(id)"
        ).execute()
        
        files_found = file_list.get('files', [])
        
        if files_found:
            # Update existing file
            file_id = files_found[0]['id']
            file_metadata = {'name': file_name, 'parents': [folder_id]} # Update metadata with parents
            media_body = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            drive_service.files().update(fileId=file_id, body=file_metadata, media_body=media_body, fields='id').execute()
            st.info(f"Updated existing file on Drive: {file_name}")
        else:
            # Create new file
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media_body = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            drive_service.files().create(body=file_metadata, media_body=media_body, fields='id').execute()
            st.info(f"Uploaded new file to Drive: {file_name}")
        return True
    except Exception as e:
        st.error(f"Error uploading {file_name} to Google Drive: {e}")
        return False

def download_model_from_drive(drive_service, folder_id, file_name, local_path):
    try:
        file_list = drive_service.files().list(
            q=f"'{folder_id}' in parents and name='{file_name}'",
            fields="files(id)"
        ).execute()

        files_found = file_list.get('files', [])
        if not files_found:
            st.warning(f"'{file_name}' not found in Google Drive folder.")
            return False

        file_id = files_found[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f"Download progress: {int(status.progress() * 100)}%") # Optional: for debugging download
        
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())
        st.info(f"Downloaded '{file_name}' from Google Drive.")
        return True
    except Exception as e:
        st.error(f"Error downloading '{file_name}' from Google Drive: {e}")
        return False

# CORRECTED FUNCTION DEFINITION
def delete_model_files_from_drive(drive_service, folder_id):
    """Deletes old model and label encoder files from the specified Google Drive folder."""
    try:
        # Search for model files in the specified folder
        file_list = drive_service.files().list(
            q=f"'{folder_id}' in parents and (name contains 'ai_model.joblib' or name contains 'label_encoder.joblib')",
            fields="files(id, name)"
        ).execute()

        files_found = file_list.get('files', [])
        if not files_found:
            st.info("No old model files found in Google Drive to delete.")
            return

        for file_item in files_found:
            drive_service.files().delete(fileId=file_item['id']).execute()
            st.info(f"Deleted old model file from Drive: {file_item['name']}")
    except Exception as e:
        # It's okay if deletion fails, especially if files don't exist or permissions are off.
        # Just log the error and continue.
        st.error(f"Error deleting old model files from Google Drive: {e}")

# --- ML Model Training and Prediction ---
def train_and_save_prediction_model():
    st.write("Preparing data for AI model training...")
    all_rounds_df = load_all_historical_rounds_from_sheet()
    
    if all_rounds_df.empty:
        st.warning("No historical data available to train the AI model.")
        # If no data, delete old models from Drive
        drive = get_drive_service()
        if drive:
            delete_model_files_from_drive(drive, MODEL_FOLDER_ID) # Pass drive object and folder ID
        return False

    # Filter out columns that are not features or target
    feature_cols = ['Card 1 Value', 'Card 2 Value', 'Card 3 Value', 'Dealer Card Value', 'Player Sum']
    target_col = 'Round Outcome'

    # Ensure all feature columns exist and are numeric, drop rows with NaN in these critical columns
    required_cols = feature_cols + [target_col]
    all_rounds_df_cleaned = all_rounds_df.dropna(subset=required_cols).copy()

    if all_rounds_df_cleaned.empty:
        st.warning("No complete historical data (after cleaning missing values) to train the AI model.")
        drive = get_drive_service()
        if drive:
            delete_model_files_from_drive(drive, MODEL_FOLDER_ID)
        return False

    X = all_rounds_df_cleaned[feature_cols]
    y = all_rounds_df_cleaned[target_col]

    # Check for sufficient unique classes in target variable
    unique_outcomes = y.unique()
    if len(unique_outcomes) < 2:
        st.warning(f"Not enough diverse outcomes in your data ({len(unique_outcomes)} found). Need at least two different outcomes (e.g., 'Over 21', 'Under 21', 'Exactly 21') to train the AI model.")
        drive = get_drive_service()
        if drive:
            delete_model_files_from_drive(drive, MODEL_FOLDER_ID)
        return False
    
    st.write(f"Training AI model with {len(X)} samples.")
    
    try:
        # Fit LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(y) # Fit on all possible outcomes in the data

        # Check if all 3 expected outcomes are present after fitting (optional but good for debugging)
        expected_outcomes = ['Player Over 21', 'Player Under 21', 'Player Exactly 21']
        if not all(outcome in label_encoder.classes_ for outcome in expected_outcomes):
            st.warning("The trained LabelEncoder does not contain all expected outcomes ('Player Over 21', 'Player Under 21', 'Player Exactly 21'). Ensure your historical data includes all of them.")
            # This warning doesn't stop training, but indicates a potential future prediction issue

        y_encoded = label_encoder.transform(y)

        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y_encoded)

        # Save model and label encoder
        model_path = 'ai_model.joblib'
        encoder_path = 'label_encoder.joblib'
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        st.success("AI model and Label Encoder saved locally.")

        # Upload to Google Drive
        drive = get_drive_service()
        if drive:
            # Delete old models first to avoid clutter
            delete_model_files_from_drive(drive, MODEL_FOLDER_ID) # Ensure this call is correct
            upload_model_to_drive(drive, MODEL_FOLDER_ID, model_path, 'application/octet-stream')
            upload_model_to_drive(drive, MODEL_FOLDER_ID, encoder_path, 'application/octet-stream')
            st.success("AI model and Label Encoder uploaded to Google Drive.")
        else:
            st.warning("Could not upload model to Google Drive due to connection error.")
            
        st.session_state.ai_model = model
        st.session_state.label_encoder = label_encoder
        st.success("AI Model training complete and loaded into session!")
        return True
    except Exception as e:
        st.error(f"AI model training failed. See messages above.")
        st.exception(e) # This will print the full traceback in the Streamlit logs
        return False

def load_prediction_model():
    if 'ai_model' in st.session_state and 'label_encoder' in st.session_state:
        st.success("AI Model loaded from session state.")
        return True
    
    st.info("Attempting to load AI Model from Google Drive...")
    drive = get_drive_service()
    if not drive:
        return False

    model_path = 'ai_model.joblib'
    encoder_path = 'label_encoder.joblib'

    # Download from Drive
    model_downloaded = download_model_from_drive(drive, MODEL_FOLDER_ID, 'ai_model.joblib', model_path)
    encoder_downloaded = download_model_from_drive(drive, MODEL_FOLDER_ID, 'label_encoder.joblib', encoder_path)

    if model_downloaded and encoder_downloaded:
        try:
            st.session_state.ai_model = joblib.load(model_path)
            st.session_state.label_encoder = joblib.load(encoder_path)
            st.success("AI Model loaded from Google Drive.")
            return True
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            return False
    else:
        st.warning("Could not load AI Model from Google Drive. Please train the model first.")
        return False

# --- Prediction Logic ---
def predict_outcome(model, label_encoder, features_df):
    try:
        probabilities = model.predict_proba(features_df)[0]
        # Create a DataFrame for display
        prob_df = pd.DataFrame({
            'Outcome': label_encoder.classes_,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False)
        return prob_df
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def analyze_patterns(historical_df):
    recent_outcomes = historical_df['Round Outcome'].tail(PREDICTION_ROUNDS_CONSIDERED).tolist()
    st.subheader("Pattern Analysis")
    predicted_by_pattern = False

    if len(recent_outcomes) >= 2:
        for pattern_name, pattern_sequence in PATTERNS_TO_WATCH.items():
            if len(recent_outcomes) >= len(pattern_sequence) and recent_outcomes[-len(pattern_sequence):] == pattern_sequence:
                if pattern_name.endswith('_U'):
                    st.write(f"Strong Pattern '{pattern_name}' observed! Prediction: **Player Under 21** for the next round.")
                    predicted_by_pattern = True
                elif pattern_name.endswith('_O'):
                    st.write(f"Strong Pattern '{pattern_name}' observed! Prediction: **Player Over 21** for the next round.")
                    predicted_by_pattern = True
                elif pattern_name.endswith('_E'):
                    st.write(f"Strong Pattern '{pattern_name}' observed! Prediction: **Player Exactly 21** for the next round.")
                    predicted_by_pattern = True
                else: # For 2-outcome patterns without explicit next round prediction
                     st.write(f"Pattern '{pattern_name}' observed in recent rounds.")
                     predicted_by_pattern = True

    # Simple bias detection if no strong pattern
    if not predicted_by_pattern and len(recent_outcomes) > 5:
        over_21_count = recent_outcomes.count('Player Over 21')
        under_21_count = recent_outcomes.count('Player Under 21')
        exactly_21_count = recent_outcomes.count('Player Exactly 21')

        total_considered = over_21_count + under_21_count + exactly_21_count

        if total_considered > 0:
            over_bias = over_21_count / total_considered
            under_bias = under_21_count / total_considered
            exactly_bias = exactly_21_count / total_considered

            if over_bias > OVER_UNDER_BIAS_THRESHOLD and over_bias > under_bias and over_bias > exactly_bias:
                st.info(f"Recent rounds show a bias towards 'Over 21' ({over_bias:.1%}). Consider 'Player Under 21' for next round.")
                predicted_by_pattern = True
            elif under_bias > OVER_UNDER_BIAS_THRESHOLD and under_bias > over_bias and under_bias > exactly_bias:
                st.info(f"Recent rounds show a bias towards 'Under 21' ({under_bias:.1%}). Consider 'Player Over 21' for next round.")
                predicted_by_pattern = True
            elif exactly_bias > OVER_UNDER_BIAS_THRESHOLD and exactly_bias > over_bias and exactly_bias > under_bias:
                st.info(f"Recent rounds show a bias towards 'Exactly 21' ({exactly_bias:.1%}).")
                predicted_by_pattern = True

    if not predicted_by_pattern:
        st.info("No strong patterns or biases observed in recent rounds.")
    
    return predicted_by_pattern


# --- Main App ---
st.set_page_config(layout="wide", page_title="Casino Card Game Tracker & Predictor")

st.title("🃏 Casino Card Game Tracker & Predictor")

# Initialize session state for AI model and encoder
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Sidebar for Model Management
with st.sidebar:
    st.header("AI Model Management")
    ai_model_status_placeholder = st.empty()
    
    if st.session_state.ai_model and st.session_state.label_encoder:
        ai_model_status_placeholder.success("AI Model Ready: ✅")
    else:
        ai_model_status_placeholder.warning("AI Model Not Ready: ❌ (Train it!)")
        
    if st.button("Train/Retrain AI Model"):
        ai_model_status_placeholder.info("Training AI model... This might take a moment.")
        with st.spinner("Training AI model... This might take a moment."):
            training_successful = train_and_save_prediction_model()
            if training_successful:
                ai_model_status_placeholder.success("AI Model Ready: ✅")
                st.rerun() # Rerun to update the status and potentially enable predictions
            else:
                ai_model_status_placeholder.error("AI Model training failed. See messages above.")
    
    # Check if model can be loaded from Drive on initial load if not in session
    if not (st.session_state.ai_model and st.session_state.label_encoder):
        if load_prediction_model():
            ai_model_status_placeholder.success("AI Model Ready: ✅")
            st.rerun() # Rerun to update the status and enable predictions

# Card Selection Inputs
st.header("Log New Round")

col1, col2, col3, col4 = st.columns(4)

with col1:
    card1_str = st.selectbox("Player Card 1", ['-'] + all_cards_list, key="p1")
    card1_value = card_values.get(card1_str[:-1], 0) if card1_str != '-' else 0

with col2:
    card2_str = st.selectbox("Player Card 2", ['-'] + all_cards_list, key="p2")
    card2_value = card_values.get(card2_str[:-1], 0) if card2_str != '-' else 0

with col3:
    card3_str = st.selectbox("Player Card 3", ['-'] + all_cards_list, key="p3")
    card3_value = card_values.get(card3_str[:-1], 0) if card3_str != '-' else 0

player_sum = card1_value + card2_value + card3_value
st.metric("Player Hand Sum", player_sum)

with col4:
    dealer_card_str = st.selectbox("Dealer Card", ['-'] + all_cards_list, key="d1")
    dealer_card_value = card_values.get(dealer_card_str[:-1], 0) if dealer_card_str != '-' else 0

st.subheader("Round Outcome")
outcome = st.radio(
    "What was the round outcome for the Player?",
    ('Player Over 21', 'Player Under 21', 'Player Exactly 21'),
    index=1 # Default to Under 21
)

if st.button("Log Round to Google Sheet"):
    if player_sum == 0 or dealer_card_value == 0:
        st.error("Please select all three player cards and the dealer's card before logging.")
    else:
        round_data = {
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Card 1': card1_str,
            'Card 1 Value': card1_value,
            'Card 2': card2_str,
            'Card 2 Value': card2_value,
            'Card 3': card3_str,
            'Card 3 Value': card3_value,
            'Dealer Card': dealer_card_str,
            'Dealer Card Value': dealer_card_value,
            'Player Sum': player_sum,
            'Round Outcome': outcome
        }
        log_round_to_sheet(round_data)

# --- Display Recent History ---
st.header("Recent Game History")
historical_df = load_all_historical_rounds_from_sheet()

if not historical_df.empty:
    st.dataframe(historical_df.tail(10), use_container_width=True) # Display last 10 rounds
else:
    st.info("No game history to display. Log a round above!")

# --- AI Model Prediction Section ---
st.header("AI Model Prediction for Current Hand")
card1_selected = card1_str != '-'
card2_selected = card2_str != '-'
card3_selected = card3_str != '-'
dealer_card_selected = dealer_card_str != '-'

ai_model_prediction_attempted = False

if card1_selected and card2_selected and card3_selected and dealer_card_selected:
    if st.session_state.ai_model and st.session_state.label_encoder:
        ai_model_prediction_attempted = True
        try:
            # Prepare features for prediction
            features = pd.DataFrame([[card1_value, card2_value, card3_value, dealer_card_value, player_sum]],
                                    columns=['Card 1 Value', 'Card 2 Value', 'Card 3 Value', 'Dealer Card Value', 'Player Sum'])
            
            probabilities_df = predict_outcome(st.session_state.ai_model, st.session_state.label_encoder, features)
            if probabilities_df is not None:
                st.subheader("AI Model's Predicted Probabilities")
                st.write("Based on your historical data, the AI model predicts the following probabilities for this hand:")
                st.dataframe(probabilities_df, hide_index=True, use_container_width=True)
                
                # Highlight highest probability outcome
                best_outcome = probabilities_df.iloc[0]['Outcome']
                best_prob = probabilities_df.iloc[0]['Probability']
                st.info(f"AI Model's Top Prediction: **{best_outcome}** with **{best_prob:.1%}** probability for the next round.")
            else:
                st.warning("Could not generate AI model prediction for this hand.")

        except ValueError as e:
            st.error(f"AI Model prediction error: {e}. This means the input data for prediction might be inconsistent with training data (e.g., non-numeric values for cards/sum).")
            ai_model_prediction_error_occurred = True
        except Exception as e:
            st.error(f"An unexpected error occurred during AI model prediction: {e}")
            ai_model_prediction_error_occurred = True
    else:
        st.info("AI Model is not loaded. Please train the model first to get predictions.")
else:
    st.info("Select all three player cards and the dealer card to see the AI Model's Prediction for this hand.")

# --- Pattern Analysis Section ---
if not historical_df.empty:
    predicted_by_pattern = analyze_patterns(historical_df)
