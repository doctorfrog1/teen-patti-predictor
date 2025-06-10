import streamlit as st
import pandas as pd
from datetime import datetime
import gspread
import os

# --- Configuration ---
PLAYER_A_FIXED_CARDS_STR = ['J♣', '10♠', '9♠'] # Player A's fixed cards (assuming they are always out of play)
PREDICTION_ROUNDS_CONSIDERED = 10 # Number of previous rounds to consider for simple prediction
STREAK_THRESHOLD = 3 # Minimum streak length to highlight
OVER_UNDER_BIAS_THRESHOLD = 0.6 # If Over/Under > 60% of rounds, show bias

# Add this PATTERNS_TO_WATCH dictionary here
PATTERNS_TO_WATCH = {
    # Existing patterns (specific longer sequences)
    'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21'],
    'UUU_O': ['Under 21', 'Under 21', 'Under 21', 'Over 21'],
    'OUOU': ['Over 21', 'Under 21', 'Over 21', 'Under 21'],
    'UOUO': ['Under 21', 'Over 21', 'Under 21', 'Over 21'],

    # Shorter, "binary-like" patterns and reversals (O=Over, U=Under)
    'OO': ['Over 21', 'Over 21'],
    'UU': ['Under 21', 'Under 21'],
    'O_U': ['Over 21', 'Under 21'],
    'U_O': ['Under 21', 'Over 21'],
    'OOU': ['Over 21', 'Over 21', 'Under 21'],
    'UUO': ['Under 21', 'Under 21', 'Over 21'],
    'OUU': ['Over 21', 'Under 21', 'Under 21'],
    'UOO': ['Under 21', 'Over 21', 'Over 21'],

    # Longer streaks
    'OOO': ['Over 21', 'Over 21', 'Over 21'],
    'UUU': ['Under 21', 'Under 21', 'Under 21'],
    'OOOO': ['Over 21', 'Over 21', 'Over 21', 'Over 21'],
    'UUUU': ['Under 21', 'Under 21', 'Under 21', 'Under 21'],

    # Alternating sequences (O=Over, U=Under)
    'Alt_O_U_O': ['Over 21', 'Under 21', 'Over 21'],
    'Alt_U_O_U': ['Under 21', 'Over 21', 'Under 21'],

    # Patterns involving "Exactly 21" (E=Exactly 21)
    'E': ['Exactly 21'],
    'EE': ['Exactly 21', 'Exactly 21'],
    'OE': ['Over 21', 'Exactly 21'],
    'UE': ['Under 21', 'Exactly 21'],
    'EO': ['Exactly 21', 'Over 21'],
    'EU': ['Exactly 21', 'Under 21'],
    'OEO': ['Over 21', 'Exactly 21', 'Over 21'],
    'UEU': ['Under 21', 'Exactly 21', 'Under 21'],
    'E_O_O': ['Exactly 21', 'Over 21', 'Over 21'],
    'E_U_U': ['Exactly 21', 'Under 21', 'Under 21'],
    'O_E_U': ['Over 21', 'Exactly 21', 'Under 21'],
    'U_E_O': ['Under 21', 'Exactly 21', 'Over 21'],
}

# ... (rest of your existing code below this) ...

# Define card values
card_values = {
    'A♠': 1, '2♠': 2, '3♠': 3, '4♠': 4, '5♠': 5, '6♠': 6, '7♠': 7, '8♠': 8, '9♠': 9, '10♠': 10, 'J♠': 11, 'Q♠': 12, 'K♠': 13,
    'A♦': 1, '2♦': 2, '3♦': 3, '4♦': 4, '5♦': 5, '6♦': 6, '7♦': 7, '8♦': 8, '9♦': 9, '10♦': 10, 'J♦': 11, 'Q♦': 12, 'K♦': 13,
    'A♣': 1, '2♣': 2, '3♣': 3, '4♣': 4, '5♣': 5, '6♣': 6, '7♣': 7, '8♣': 8, '9♣': 9, '10♣': 10, 'J♣': 11, 'Q♣': 12, 'K♣': 13,
    'A♥': 1, '2♥': 2, '3♥': 3, '4♥': 4, '5♥': 5, '6♥': 6, '7♥': 7, '8♥': 8, '9♥': 9, '10♥': 10, 'J♥': 11, 'Q♥': 12, 'K♥': 13
}

ALL_CARDS = list(card_values.keys())

# --- Session State Initialization ---
if 'rounds' not in st.session_state:
    st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

if 'current_deck_id' not in st.session_state:
    st.session_state.current_deck_id = 1 # Start with Deck 1

if 'played_cards' not in st.session_state:
    st.session_state.played_cards = set()

# Function to get gspread client from Streamlit secrets
@st.cache_resource
def get_gspread_client():
    try:
        # st.secrets.gcp_service_account directly accesses the [gcp_service_account] section
        gc = gspread.service_account_from_dict(st.secrets.gcp_service_account)
        return gc
    except Exception as e:
        st.error(f"Error loading Google Sheets credentials: {e}. Please ensure st.secrets are configured.")
        st.stop() # Stop the app if credentials are not loaded
        return None

# --- Functions ---
def load_rounds():
    gc = get_gspread_client()
    if not gc:
        # Fallback if credentials failed, though st.stop() should prevent reaching here
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
        # Also ensure played_cards is cleared if we can't load history
        st.session_state.played_cards = set(PLAYER_A_FIXED_CARDS_STR) # Only Player A's cards initially
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log") # <--- ENSURE THIS MATCHES YOUR SHEET NAME
        worksheet = spreadsheet.worksheet("Sheet1") # <--- ENSURE THIS MATCHES YOUR SHEET TAB NAME

        data = worksheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if 'Deck_ID' not in df.columns:
                df['Deck_ID'] = 1
            df['Deck_ID'] = df['Deck_ID'].astype(int)
            st.session_state.rounds = df
        else:
            st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Google Sheet 'Casino Card Game Log' not found. Please ensure the name is correct and it's shared with the service account.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    except Exception as e:
        st.error(f"Error loading rounds from Google Sheet: {e}. Starting with empty history.")
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])

    # --- REVISED played_cards and current_deck_id Initialization Logic ---
    # This logic ensures played_cards is correctly set based on the *current* deck in st.session_state.rounds

    # Set current_deck_id:
    # If no rounds loaded, start at Deck 1. Otherwise, use the highest Deck_ID from loaded data.
    if st.session_state.rounds.empty:
        st.session_state.current_deck_id = 1
    else:
        st.session_state.current_deck_id = st.session_state.rounds['Deck_ID'].max()

    # Initialize played_cards for the *current* deck
    st.session_state.played_cards = set() # Always start fresh for the current deck's played cards

    # Add cards played in the current deck from the history
    if not st.session_state.rounds.empty:
        current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
        for _, row in current_deck_rounds.iterrows():
            st.session_state.played_cards.add(row['Card1'])
            st.session_state.played_cards.add(row['Card2'])
            st.session_state.played_cards.add(row['Card3'])

    # Always add Player A's fixed cards (they are never available)
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

    st.write(f"DEBUG FINAL played_cards after load_rounds logic: {st.session_state.played_cards}") # NEW DEBUG
    
def save_rounds():
    gc = get_gspread_client()
    if not gc:
        st.warning("Cannot save rounds: Google Sheets client not available.")
        return

    try:
        spreadsheet = gc.open("Casino Card Game Log") # <--- ENSURE THIS MATCHES YOUR SHEET NAME
        worksheet = spreadsheet.worksheet("Sheet1") # <--- ENSURE THIS MATCHES YOUR SHEET TAB NAME

        # Convert DataFrame to a list of lists (including header)
        # Use .astype(str) for all columns to ensure consistent string writing to Sheets
        data_to_write = [st.session_state.rounds.columns.tolist()] + st.session_state.rounds.astype(str).values.tolist()

        # Clear existing content and write DataFrame from A1
        # This ensures the sheet is always a fresh copy of your DataFrame
        worksheet.clear()
        worksheet.update('A1', data_to_write)
        # st.success("Rounds saved to Google Sheet.") # You can add this back for feedback

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Cannot save: Google Sheet 'Casino Card Game Log' not found. Please create the sheet and share it correctly.")
    except Exception as e:
        st.error(f"Error saving rounds to Google Sheet: {e}")

def get_current_streak(df):
    """Calculates the current streak of 'Over' or 'Under'."""
    if df.empty:
        return None, 0

    current_outcome = df.iloc[-1]['Outcome']
    streak_count = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]['Outcome'] == current_outcome:
            streak_count += 1
        else:
            break
    return current_outcome, streak_count
    
def predict_next_outcome_from_pattern(df_all_rounds, pattern_sequence):
    """
    Analyzes historical data to predict the next outcome after a given pattern.

    Args:
        df_all_rounds (pd.DataFrame): The full DataFrame of all historical rounds.
        pattern_sequence (list): The sequence of outcomes to look for (e.g., ['Over 21', 'Over 21']).

    Returns:
        tuple: (most_likely_outcome, confidence_percentage) or (None, 0) if no data.
    """
    if df_all_rounds.empty or not pattern_sequence:
        return None, 0

    next_outcomes = []
    pattern_len = len(pattern_sequence)

    # Iterate through all rounds, checking in each deck context
    # Group by Deck_ID to prevent patterns from crossing deck boundaries
    for deck_id, deck_df in df_all_rounds.groupby('Deck_ID'):
        outcomes_in_deck = deck_df['Outcome'].tolist()
        
        for i in range(len(outcomes_in_deck) - pattern_len): # -pattern_len because we need a subsequent outcome
            if outcomes_in_deck[i : i + pattern_len] == pattern_sequence:
                # If the pattern is found, check the very next outcome
                if (i + pattern_len) < len(outcomes_in_deck): # Ensure there IS a next outcome
                    next_outcomes.append(outcomes_in_deck[i + pattern_len])
    
    if not next_outcomes:
        return None, 0
    
    # Count occurrences of each next outcome
    outcome_counts = pd.Series(next_outcomes).value_counts()
    most_likely_outcome = outcome_counts.index[0]
    confidence_percentage = (outcome_counts.iloc[0] / len(next_outcomes)) * 100

    return most_likely_outcome, confidence_percentage
    
def find_patterns(df, patterns_to_watch):
    """
    Detects predefined sequences (patterns) in the outcomes of a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with an 'Outcome' column.
        patterns_to_watch (dict): A dictionary where keys are pattern names
                                  and values are lists of outcomes, e.g.,
                                  {'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21']}
    Returns:
        dict: Counts of how many times each pattern was found.
    """
    pattern_counts = {name: 0 for name in patterns_to_watch.keys()}
    outcomes = df['Outcome'].tolist()

    for pattern_name, pattern_sequence in patterns_to_watch.items():
        pattern_len = len(pattern_sequence)
        for i in range(len(outcomes) - pattern_len + 1):
            if outcomes[i:i+pattern_len] == pattern_sequence:
                pattern_counts[pattern_name] += 1
    return pattern_counts

def reset_deck():
    st.session_state.current_deck_id += 1
    st.write(f"DEBUG: New Deck button clicked. current_deck_id is now {st.session_state.current_deck_id}") # NEW DEBUG
    st.session_state.played_cards = set() # This clears played cards for the NEW deck

    # Player A's fixed cards are re-added immediately for the new deck
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

    st.success(f"Starting New Deck: Deck {st.session_state.current_deck_id}. Played cards reset for this deck.")
    st.experimental_rerun() # Force a rerun to apply the new state immediately

# --- Load data on app startup ---
load_rounds()

st.title("Casino Card Game Tracker & Predictor")

# --- Deck ID and Reset ---
st.sidebar.header(f"Current Deck: ID {st.session_state.current_deck_id}")
if st.sidebar.button("New Deck (Reset Learning)"):
    reset_deck()
    st.rerun() # Rerun to update the available cards list and clear display

# --- Card Input Section ---
st.header("Enter Round Details")

# Filter available cards by removing played cards and Player A's fixed cards
available_cards_for_selection = [card for card in ALL_CARDS if card not in st.session_state.played_cards]


card1 = st.selectbox("Select Card 1", available_cards_for_selection, key="card1_select")
card2 = st.selectbox("Select Card 2", [c for c in available_cards_for_selection if c != card1], key="card2_select")
card3 = st.selectbox("Select Card 3", [c for c in available_cards_for_selection if c != card1 and c != card2], key="card3_select")

# Calculate total
if card1 and card2 and card3: # Ensure all cards are selected before calculating
    total = card_values[card1] + card_values[card2] + card_values[card3]
    st.write(f"**Calculated Total:** {total}")

    # Determine outcome
    outcome = ""
    if total > 21:
        outcome = "Over 21"
        st.success("Result: Over 21")
    elif total < 21:
        outcome = "Under 21"
        st.info("Result: Under 21")
    else:
        outcome = "Exactly 21"
        st.warning("Result: Exactly 21")

    # Add round button
    if st.button("Add Round"):
        timestamp = datetime.now()
        round_id = len(st.session_state.rounds) + 1
        new_round = {
            'Timestamp': timestamp,
            'Round_ID': round_id,
            'Card1': card1,
            'Card2': card2,
            'Card3': card3,
            'Sum': total,
            'Outcome': outcome,
            'Deck_ID': st.session_state.current_deck_id
        }
        st.session_state.rounds = pd.concat([st.session_state.rounds, pd.DataFrame([new_round])], ignore_index=True)
        
        # Add cards to played_cards set for the current deck
        st.session_state.played_cards.add(card1)
        st.session_state.played_cards.add(card2)
        st.session_state.played_cards.add(card3)

        save_rounds()
        st.rerun() # Rerun to update displays and available cards
else:
    st.write("Please select all three cards to calculate the total and add the round.")

## Real-time Insights

### Current Streak

if not st.session_state.rounds.empty:
    current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id].copy()
    if not current_deck_rounds.empty:
        streak_outcome, streak_length = get_current_streak(current_deck_rounds)
        if streak_length >= STREAK_THRESHOLD:
            st.markdown(f"**Current Streak:** 🔥 {streak_length}x **{streak_outcome}** in a row! 🔥")
        elif streak_length > 0:
            st.write(f"**Current Streak:** {streak_length}x {streak_outcome}")
    else:
        st.write("No rounds played in the current deck yet to determine a streak.")
else:
    st.write("No rounds played yet.")

### Daily Tendency

if not st.session_state.rounds.empty:
    today_date = datetime.now().date()
    daily_rounds = st.session_state.rounds[st.session_state.rounds['Timestamp'].dt.date == today_date]

    if not daily_rounds.empty:
        over_count = daily_rounds[daily_rounds['Outcome'] == 'Over 21'].shape[0]
        under_count = daily_rounds[daily_rounds['Outcome'] == 'Under 21'].shape[0]
        total_daily_outcomes = over_count + under_count # Exclude 'Exactly 21' for bias calculation if desired

        if total_daily_outcomes > 0:
            over_percentage = over_count / total_daily_outcomes
            under_percentage = under_count / total_daily_outcomes

            st.write(f"**Today's Outcomes (Deck {st.session_state.current_deck_id}):**")
            st.write(f"- Over 21: {over_count} ({over_percentage:.1%})")
            st.write(f"- Under 21: {under_count} ({under_percentage:.1%})")
            st.write(f"- Exactly 21: {daily_rounds[daily_rounds['Outcome'] == 'Exactly 21'].shape[0]}")

            if over_percentage > OVER_UNDER_BIAS_THRESHOLD:
                st.markdown(f"📈 **Today's Trend:** Leaning towards **Over 21**!")
            elif under_percentage > OVER_UNDER_BIAS_THRESHOLD:
                st.markdown(f"📉 **Today's Trend:** Leaning towards **Under 21**!")
            else:
                st.write("📊 **Today's Trend:** Fairly balanced between Over and Under.")
        else:
            st.write("No 'Over 21' or 'Under 21' outcomes recorded for today yet.")
    else:
        st.write("No rounds recorded for today yet.")
else:
    st.write("No historical rounds to analyze daily tendency.")
    
# ... (After Daily Tendency section) ...

st.header("Observed Patterns (Current Deck)")

if not st.session_state.rounds.empty:
    current_deck_rounds_for_patterns = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id].copy()
    if not current_deck_rounds_for_patterns.empty:
        # Call the find_patterns function you added in Step 2
        pattern_counts = find_patterns(current_deck_rounds_for_patterns, PATTERNS_TO_WATCH)

        found_any_pattern = False
        for pattern_name, count in pattern_counts.items():
            if count > 0:
                st.write(f"- `{pattern_name}`: Found **{count}** time(s)")
                found_any_pattern = True

        if not found_any_pattern:
            st.write("No defined patterns observed in the current deck yet.")
    else:
        st.write("No rounds played in the current deck to find patterns.")
else:
    st.write("No historical rounds to find patterns.")



## Prediction Module

st.header("Next Round Prediction")

if not st.session_state.rounds.empty:
    # --- Pattern-Based Prediction Attempt ---
    predicted_by_pattern = False
    pattern_prediction_outcome = None
    pattern_prediction_confidence = 0

    current_deck_outcomes = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]['Outcome'].tolist()

    if len(current_deck_outcomes) >= 2: # Need at least 2 outcomes to check for any pattern end
        # Check for the most recent completed pattern that can predict
        # Iterate through patterns in reverse order of length to prioritize longer, more specific ones
        sorted_patterns = sorted(PATTERNS_TO_WATCH.items(), key=lambda item: len(item[1]), reverse=True)

        for pattern_name, pattern_sequence in sorted_patterns:
            pattern_len = len(pattern_sequence)
            if len(current_deck_outcomes) >= pattern_len and \
               current_deck_outcomes[-pattern_len:] == pattern_sequence:
                # This pattern just completed! Now predict based on it
                
                # Pass the *entire* rounds DataFrame for historical analysis
                outcome, confidence = predict_next_outcome_from_pattern(st.session_state.rounds, pattern_sequence)
                
                if outcome:
                    pattern_prediction_outcome = outcome
                    pattern_prediction_confidence = confidence
                    st.write(f"Based on pattern `{pattern_name}` (last {pattern_len} rounds):")
                    st.markdown(f"**Prediction:** ➡️ **{pattern_prediction_outcome}** (Confidence: {pattern_prediction_confidence:.1f}%)")
                    predicted_by_pattern = True
                    break # Stop at the first (longest) matching pattern
    
    # --- Fallback to Simple Frequency-Based Prediction if no pattern was used ---
    if not predicted_by_pattern:
        recent_rounds = st.session_state.rounds[
            st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id
        ].tail(PREDICTION_ROUNDS_CONSIDERED)

        if not recent_rounds.empty:
            outcome_counts = recent_rounds['Outcome'].value_counts()
            
            # Exclude 'Exactly 21' from prediction logic if it's not relevant for Over/Under betting
            if 'Exactly 21' in outcome_counts.index:
                outcome_counts = outcome_counts.drop(labels='Exactly 21', errors='ignore')

            if not outcome_counts.empty:
                predicted_outcome = outcome_counts.index[0]
                st.write(f"Based on last {len(recent_rounds)} rounds (current deck):")
                st.markdown(f"**Prediction:** ➡️ **{predicted_outcome}**")
            else:
                st.write("Not enough 'Over 21' or 'Under 21' outcomes in recent rounds for a simple prediction.")
        else:
            st.write("Not enough rounds played in current deck for a prediction.")
else:
    st.write("No historical rounds available for prediction.")
    
### Full Round History

st.header("Round History (Current Deck)")
if not st.session_state.rounds.empty:
    current_deck_history = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
    if not current_deck_history.empty:
        st.dataframe(current_deck_history.set_index('Round_ID'))
    else:
        st.write("No rounds played in the current deck yet.")
else:
    st.write("No rounds recorded yet.")
