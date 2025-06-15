import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import gspread
from gspread.exceptions import SpreadsheetNotFound
import os
import joblib

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# --- Configuration ---
MODEL_FOLDER_ID = "1CZepfjRZxWV_wfmEQuZLnbj9H2yAS9Ac"
PLAYER_A_FIXED_CARDS_STR = {'J♣', '10♠', '9♠'}
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
    'O_O': ['Over 21', 'Over 21'],
    'U_U': ['Under 21', 'Under 21'],
    'OUU': ['Over 21', 'Under 21', 'Under 21'],
    'UOO': ['Under 21', 'Over 21', 'Over 21'],
    'OUO': ['Over 21', 'Under 21', 'Over 21'],
    'UOU': ['Under 21', 'Over 21', 'Under 21'],
    'OOOU': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUUO': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OOUU': ['Over 21', 'Over 21', 'Under 21', 'Under 21'],
    'UUOO': ['Under 21', 'Under 21', 'Over 21', 'Over 21'],
}



# --- Google Sheets Authentication and Data Loading ---

@st.cache_resource(ttl="1h")
def get_gspread_client():
    try:
        # Load credentials from Streamlit secrets
        gcp_service_account_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            gcp_service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {e}")
        st.stop() # Stop the app if authentication fails

def get_drive_service():
    try:
        gcp_service_account_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            gcp_service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Error authenticating with Google Drive: {e}")
        return None


def load_data(client):
    try:
        spreadsheet_name = st.secrets["private_gsheets_url"].split('/')[5]
        # Attempt to open by name first
        sh = client.open(spreadsheet_name)
    except SpreadsheetNotFound:
        # If not found by name, try opening by URL
        sh = client.open_by_url(st.secrets["private_gsheets_url"])
    
    worksheet = sh.worksheet("Sheet1") # Or your specific worksheet name
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    # Data Cleaning and Type Conversion
    if not df.empty:
        # Convert 'Date' to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Convert card columns to string to handle mixed types if any
        for col in ['Card 1', 'Card 2', 'Card 3']:
            df[col] = df[col].astype(str)
        # Convert numeric columns, coerce errors will turn non-numeric into NaN
        for col in ['Round', 'Player A Cards Sum', 'Player B Cards Sum', 'Player C Cards Sum', 'Player D Cards Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential numerical conversions failed
        df.dropna(subset=['Round', 'Player A Cards Sum'], inplace=True)
        
        # Ensure 'Outcome' column exists
        if 'Outcome' not in df.columns:
            df['Outcome'] = '' # Add it if missing

        # Ensure 'Deck_ID' exists and is numeric (or handled as string if preferred)
        if 'Date' in df.columns and not df['Date'].isnull().all():
            if 'Deck_ID' not in df.columns:
                df['Deck_ID'] = df.groupby('Date').ngroup() + 1 # Simple sequential ID per date
            else:
                # Try to convert to numeric, if fails, keep as string or generate new
                df['Deck_ID'] = pd.to_numeric(df['Deck_ID'], errors='coerce').fillna(df.groupby('Date').ngroup() + 1)
        else:
            st.warning("Date column is missing or invalid. Generating simple Deck_ID based on row index.")
            df['Deck_ID'] = (df.index // PREDICTION_ROUNDS_CONSIDERED) + 1 # Fallback if Date is problematic
            
    return df

def update_data(client, new_data):
    try:
        spreadsheet_name = st.secrets["private_gsheets_url"].split('/')[5]
        sh = client.open(spreadsheet_name)
        worksheet = sh.worksheet("Sheet1")
        
        # Append new data
        worksheet.append_row(new_data)
        st.success("Data recorded successfully!")
        st.experimental_rerun() # Rerun to refresh the data display
    except Exception as e:
        st.error(f"Error updating Google Sheet: {e}")

# --- AI Model Training and Prediction ---

# Global variable to store the model and encoder
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None


def save_model_to_drive(drive_service, model, le, folder_id):
    try:
        # Save model and encoder to BytesIO objects
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        le_buffer = io.BytesIO()
        joblib.dump(le, le_buffer)
        le_buffer.seek(0)

        # Upload/Update model file
        model_file_metadata = {
            'name': 'logistic_regression_model.joblib',
            'parents': [folder_id],
            'mimeType': 'application/octet-stream'
        }
        
        # Check if file exists to update or create
        response = drive_service.files().list(
            q=f"name='logistic_regression_model.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()
        
        files = response.get('files', [])
        if files:
            file_id = files[0]['id']
            media_body = MediaFileUpload(
                filename='logistic_regression_model.joblib', # Dummy filename for MediaFileUpload
                mimetype='application/octet-stream',
                resumable=True,
                body=model_buffer
            )
            drive_service.files().update(fileId=file_id, media_body=media_body).execute()
            st.toast("AI model updated on Google Drive.")
        else:
            media_body = MediaFileUpload(
                filename='logistic_regression_model.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=model_buffer
            )
            drive_service.files().create(body=model_file_metadata, media_body=media_body, fields='id').execute()
            st.toast("AI model saved to Google Drive.")

        # Upload/Update LabelEncoder file
        le_file_metadata = {
            'name': 'label_encoder.joblib',
            'parents': [folder_id],
            'mimeType': 'application/octet-stream'
        }

        response = drive_service.files().list(
            q=f"name='label_encoder.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()

        files = response.get('files', [])
        if files:
            file_id = files[0]['id']
            media_body = MediaFileUpload(
                filename='label_encoder.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=le_buffer
            )
            drive_service.files().update(fileId=file_id, media_body=media_body).execute()
            st.toast("LabelEncoder updated on Google Drive.")
        else:
            media_body = MediaFileUpload(
                filename='label_encoder.joblib',
                mimetype='application/octet-stream',
                resumable=True,
                body=le_buffer
            )
            drive_service.files().create(body=le_file_metadata, media_body=media_body, fields='id').execute()
            st.toast("LabelEncoder saved to Google Drive.")

    except Exception as e:
        st.error(f"Error saving model/encoder to Google Drive: {e}")

def load_model_from_drive(drive_service, folder_id):
    model = None
    le = None
    try:
        # Load model file
        model_response = drive_service.files().list(
            q=f"name='logistic_regression_model.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()
        
        model_files = model_response.get('files', [])
        if model_files:
            file_id = model_files[0]['id']
            request = drive_service.files().get_media(fileId=file_id)
            model_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(model_buffer, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            model_buffer.seek(0)
            model = joblib.load(model_buffer)
            st.toast("AI model loaded from Google Drive.")
        else:
            st.warning("AI model file not found on Google Drive.")

        # Load LabelEncoder file
        le_response = drive_service.files().list(
            q=f"name='label_encoder.joblib' and '{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id)'
        ).execute()

        le_files = le_response.get('files', [])
        if le_files:
            file_id = le_files[0]['id']
            request = drive_service.files().get_media(fileId=file_id)
            le_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(le_buffer, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            le_buffer.seek(0)
            le = joblib.load(le_buffer)
            st.toast("LabelEncoder loaded from Google Drive.")
        else:
            st.warning("LabelEncoder file not found on Google Drive.")

    except Exception as e:
        st.error(f"Error loading model/encoder from Google Drive: {e}")
    return model, le

def train_ai_model(df):
    st.info("Training AI Model... This may take a moment.")
    print("Starting train_ai_model function (for sequence prediction)...")

    # NEW DEBUG PRINT: Check columns before sorting
    print(f"DEBUG (TRAINING): Columns in df received by train_ai_model: {df.columns.tolist()}")

    # Check if 'Deck_ID' actually exists before sorting
    if 'Deck_ID' not in df.columns:
        st.error("Error: 'Deck_ID' column is missing in the data. Cannot train AI model. Please ensure your Google Sheet has data, especially a 'Date' column and is not empty.")
        return None, None # Exit early if critical column is missing

    df_sorted = df.sort_values(by=['Deck_ID', 'Round']).copy()

    # Ensure LabelEncoder is fitted on all unique outcomes that exist in the data
    # This prevents issues with unseen labels during transformation
    le = LabelEncoder()
    # Fit on all unique outcomes in the entire historical data
    le.fit(df_sorted['Outcome'].unique())
    print(f"DEBUG (TRAINING): LabelEncoder classes after fit: {le.classes_.tolist()}")

    df_sorted['Outcome_Encoded'] = le.transform(df_sorted['Outcome'])

    # Create lagged features
    lag_features = []
    for i in range(1, PREDICTION_ROUNDS_CONSIDERED + 1):
        col_name = f'Outcome_Lag{i}'
        df_sorted[col_name] = df_sorted.groupby('Deck_ID')['Outcome_Encoded'].shift(i)
        lag_features.append(col_name)

    df_train = df_sorted.dropna(subset=lag_features).copy()

    if df_train.empty:
        st.warning("Not enough data to train the AI model after creating lagged features. Please record more rounds.")
        return None, None

    X = df_train[lag_features]
    y_encoded = df_train['Outcome_Encoded']

    if y_encoded.nunique() < 2:
        st.warning("Not enough unique outcomes in the training data to train a classification model. Need at least 2 distinct outcomes.")
        return None, None

    print(f"DEBUG (TRAINING): Starting Logistic Regression training for sequence prediction with {len(X)} samples.")
    print(f"DEBUG (TRAINING): Shape of X (features): {X.shape}")
    print(f"DEBUG (TRAINING): Columns of X (features): {X.columns.tolist()}")
    print(f"DEBUG (TRAINING): Number of unique outcomes in y_encoded: {y_encoded.nunique()}")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y_encoded)

    print(f"DEBUG: Model fitted successfully for sequence prediction.")
    if hasattr(model, 'feature_names_in_'):
        print(f"DEBUG (TRAINING): Model feature names learned: {model.feature_names_in_.tolist()}")
    else:
        print(f"DEBUG (TRAINING): Model has no feature_names_in_ attribute.")

    st.success("AI Model trained successfully!")
    return model, le


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Teen Patti Tracker & Predictor")

st.title("🃏 Teen Patti Tracker & Predictor")

# Sidebar for AI Model Training
with st.sidebar:
    st.header("AI Model Management")
    st.write("Train or retrain the AI model based on your recorded game history. This is required for AI predictions.")

    client = get_gspread_client()
    drive_service = get_drive_service() # Get Drive service here

    if st.button("Train/Retrain AI Model"):
        # Load all historical data for training
        all_rounds_df = load_data(client)
        
        if all_rounds_df.empty:
            st.warning("No data available to train the AI model. Please record some rounds first.")
            st.session_state.ai_model = None
            st.session_state.label_encoder = None
        else:
            with st.spinner("Training model..."):
                model, label_encoder = train_ai_model(all_rounds_df.copy()) # Pass a copy
                if model and label_encoder:
                    st.session_state.ai_model = model
                    st.session_state.label_encoder = label_encoder
                    save_model_to_drive(drive_service, model, label_encoder, MODEL_FOLDER_ID)
                else:
                    st.error("AI Model training failed.")
    
    # Load model on app start or if not in session state
    if st.session_state.ai_model is None or st.session_state.label_encoder is None:
        with st.spinner("Loading AI Model from Drive (if available)..."):
            loaded_model, loaded_le = load_model_from_drive(drive_service, MODEL_FOLDER_ID)
            if loaded_model and loaded_le:
                st.session_state.ai_model = loaded_model
                st.session_state.label_encoder = loaded_le
            else:
                st.info("No AI model found on Drive. Please train it using the button above.")

# Current Hand Input
st.header("Current Hand Details")

col1, col2, col3 = st.columns(3)

with col1:
    card1 = st.selectbox("Card 1", options=['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'], index=None)
with col2:
    card2 = st.selectbox("Card 2", options=['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'], index=None)
with col3:
    card3 = st.selectbox("Card 3", options=['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'], index=None)

# Mapping card values to numeric for sum calculation
card_values = {
    'A': 11, 'K': 10, 'Q': 10, 'J': 10, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
}

player_a_sum = None
if card1 and card2 and card3:
    player_a_sum = card_values[card1] + card_values[card2] + card_values[card3]
    st.write(f"Player A's Cards Sum: **{player_a_sum}**")

    # Determine initial outcome based on sum
    initial_outcome = "Over 21" if player_a_sum > 21 else "Under 21"
    st.write(f"Initial Outcome based on sum: **{initial_outcome}**")
else:
    st.info("Select all three cards to calculate Player A's sum and initial outcome.")

st.header("Record Game Round")

# Fetch latest round and deck ID to pre-fill
all_rounds_df = load_data(client) # Reload data here to get the most up-to-date info
latest_deck_id = 1
latest_round = 0

if not all_rounds_df.empty:
    latest_deck_id = all_rounds_df['Deck_ID'].max()
    latest_round_in_deck = all_rounds_df[all_rounds_df['Deck_ID'] == latest_deck_id]['Round'].max()
    latest_round = latest_round_in_deck if pd.notna(latest_round_in_deck) else 0

    # If the last round was 15 for the latest deck, increment deck ID and reset round to 1
    if latest_round >= 15:
        latest_deck_id += 1
        latest_round = 1
    else:
        latest_round += 1
else:
    st.info("No historical data found. Starting a new game.")


# Input fields for recording a new round
with st.form("record_round_form"):
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        date_input = st.date_input("Date", value=datetime.now().date())
        current_deck_id = st.number_input("Deck ID", value=int(latest_deck_id), min_value=1, step=1)
        player_b_sum = st.number_input("Player B Cards Sum", min_value=0, max_value=33, value=0)
        player_d_sum = st.number_input("Player D Cards Sum", min_value=0, max_value=33, value=0)

    with col_rec2:
        current_round_num = st.number_input("Round Number", value=int(latest_round), min_value=1, max_value=15, step=1)
        player_c_sum = st.number_input("Player C Cards Sum", min_value=0, max_value=33, value=0)
        
        # Outcome selection (using initial_outcome as default if available)
        default_outcome_index = 0
        if player_a_sum is not None:
            default_outcome_index = ['Under 21', 'Over 21'].index(initial_outcome) if initial_outcome in ['Under 21', 'Over 21'] else 0
        outcome = st.selectbox("Outcome", options=['Under 21', 'Over 21'], index=default_outcome_index)
    
    submitted = st.form_submit_button("Record Round")
    if submitted:
        if card1 is None or card2 is None or card3 is None:
            st.error("Please select all three cards for Player A before recording the round.")
        else:
            new_round_data = [
                date_input.strftime("%Y-%m-%d"),
                current_deck_id,
                current_round_num,
                card1, card2, card3,
                player_a_sum,
                player_b_sum,
                player_c_sum,
                player_d_sum,
                outcome
            ]
            update_data(client, new_round_data)

st.header("Historical Data")
st.dataframe(all_rounds_df.sort_values(by=['Date', 'Deck_ID', 'Round'], ascending=[False, False, False]), use_container_width=True)

# --- Pattern Recognition and AI Prediction ---
st.header("Prediction for Next Round (AI & Patterns)")

current_deck_outcomes = []
if not all_rounds_df.empty:
    current_deck_data = all_rounds_df[all_rounds_df['Deck_ID'] == latest_deck_id].sort_values(by='Round')
    current_deck_outcomes = current_deck_data['Outcome'].tolist()

predicted_by_pattern = False

# Pattern Matching
if len(current_deck_outcomes) > 0:
    with st.expander("🔍 Pattern Recognition"):
        found_pattern = False
        for pattern_name, pattern_sequence in PATTERNS_TO_WATCH.items():
            if len(current_deck_outcomes) >= len(pattern_sequence):
                last_n_outcomes = current_deck_outcomes[-len(pattern_sequence):]
                if last_n_outcomes == pattern_sequence:
                    st.success(f"📈 Strong Pattern Detected: **{pattern_name}**")
                    st.write(f"Last outcomes: {last_n_outcomes}")
                    # You could add logic here to suggest the next outcome based on the pattern
                    found_pattern = True
                    predicted_by_pattern = True
        if not found_pattern:
            st.info("No strong patterns detected in the recent outcomes.")

# AI Model's Prediction
ai_model_prediction_attempted = False
if card1 and card2 and card3: # Only attempt AI prediction if current hand cards are selected
    ai_model_prediction_attempted = True

if ai_model_prediction_attempted and st.session_state.ai_model and st.session_state.label_encoder:
    with st.expander("🤖 AI Model's Prediction"):
        # The AI model needs enough historical data to make a prediction
        if len(current_deck_outcomes) < PREDICTION_ROUNDS_CONSIDERED:
            st.info(f"AI Model needs at least {PREDICTION_ROUNDS_CONSIDERED} past outcomes in the current deck to make a prediction. Current outcomes: {len(current_deck_outcomes)}.")
            # st.stop() # Removed st.stop() to allow the app to continue for other sections
        else:
            try:
                # Get the last PREDICTION_ROUNDS_CONSIDERED outcomes from the current deck
                recent_outcomes_for_lags = current_deck_outcomes[-PREDICTION_ROUNDS_CONSIDERED:]
                
                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): PREDICTION_ROUNDS_CONSIDERED: {PREDICTION_ROUNDS_CONSIDERED}")
                print(f"DEBUG (PREDICTION): Length of current_deck_outcomes: {len(current_deck_outcomes)}")
                print(f"DEBUG (PREDICTION): recent_outcomes_for_lags: {recent_outcomes_for_lags}")
                
                # Encode these outcomes using the trained label encoder
                recent_outcomes_encoded = st.session_state.label_encoder.transform(recent_outcomes_for_lags)

                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): Length of recent_outcomes_encoded: {len(recent_outcomes_encoded)}")
                print(f"DEBUG (PREDICTION): Encoded recent outcomes: {recent_outcomes_encoded}")
                print(f"DEBUG (PREDICTION): LabelEncoder classes in session state: {st.session_state.label_encoder.classes_.tolist()}")

                # Create a DataFrame for prediction matching the training features' structure
                prediction_features_dict = {}
                for i in range(PREDICTION_ROUNDS_CONSIDERED):
                    prediction_features_dict[f'Outcome_Lag{i+1}'] = [recent_outcomes_encoded[PREDICTION_ROUNDS_CONSIDERED - 1 - i]]

                X_predict = pd.DataFrame(prediction_features_dict)
                
                # DEBUG PRINTS
                print(f"DEBUG (PREDICTION): Shape of X_predict (features for prediction): {X_predict.shape}")
                print(f"DEBUG (PREDICTION): Columns of X_predict: {X_predict.columns.tolist()}")
                if hasattr(st.session_state.ai_model, 'feature_names_in_'):
                    print(f"DEBUG (PREDICTION): Model loaded feature names expected: {st.session_state.ai_model.feature_names_in_.tolist()}")
                else:
                    print(f"DEBUG (PREDICTION): Loaded model has no feature_names_in_ attribute.")

                predicted_encoded_outcome = st.session_state.ai_model.predict(X_predict)
                predicted_outcome_ai = st.session_state.label_encoder.inverse_transform(predicted_encoded_outcome)[0]

                # --- NEW DEBUG PRINTS FOR PROBABILITIES AND INDEXING ---
                # Call predict_proba once and store it
                proba_output = st.session_state.ai_model.predict_proba(X_predict)
                print(f"DEBUG (PREDICTION): Raw probabilities output shape: {proba_output.shape}")
                probabilities = proba_output[0] # Get the probabilities for the single prediction

                # Calculate the index and check its value
                index_for_confidence = st.session_state.label_encoder.transform([predicted_outcome_ai])[0]
                print(f"DEBUG (PREDICTION): Index calculated for confidence: {index_for_confidence}")
                print(f"DEBUG (PREDICTION): Size of probabilities array for confidence: {probabilities.shape[0]}")
                # --- END NEW DEBUG PRINTS ---

                confidence_ai = probabilities[index_for_confidence] * 100

                st.markdown(f"➡️ **{predicted_outcome_ai}** (Confidence: {confidence_ai:.1f}%)")
                st.caption(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} outcomes in the current deck.")

                prob_df = pd.DataFrame({
                    'Outcome': st.session_state.label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            except ValueError as e:
                st.error(f"AI Model prediction error: {e}. Ensure historical outcomes are consistent with model training and the LabelEncoder.")
                ai_model_prediction_error_occurred = True
            except Exception as e:
                st.error(f"An unexpected error occurred during AI model prediction: {e}")
                ai_model_prediction_error_occurred = True
else:
    # This else block covers cases where cards are not selected or model is not ready
    if not (card1 and card2 and card3):
        st.info("Select all three cards to see the AI Model's Prediction for this hand.")
    elif not (st.session_state.ai_model and st.session_state.label_encoder):
        st.info("AI Model is not loaded. Please train the model first to get predictions.")
