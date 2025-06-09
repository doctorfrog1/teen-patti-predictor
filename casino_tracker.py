import streamlit as st
import pandas as pd
from datetime import datetime
import os

# --- Configuration ---
LOG_FILE = 'round_log.csv'
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

# --- Functions ---
def load_rounds():
    """Loads round data from CSV."""
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, parse_dates=['Timestamp'])
        # Ensure 'Deck_ID' column exists, add if not (for old logs)
        if 'Deck_ID' not in df.columns:
            df['Deck_ID'] = 1 # Assign default Deck_ID 1 to old entries
        st.session_state.rounds = df
    else:
        st.session_state.rounds = pd.DataFrame(columns=['Timestamp', 'Round_ID', 'Card1', 'Card2', 'Card3', 'Sum', 'Outcome', 'Deck_ID'])
    
    # Initialize played_cards from the current deck's history
    # This ensures consistency if the app is re-run and a deck is ongoing
    if not st.session_state.rounds.empty:
        current_deck_rounds = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]
        st.session_state.played_cards = set()
        for _, row in current_deck_rounds.iterrows():
            st.session_state.played_cards.add(row['Card1'])
            st.session_state.played_cards.add(row['Card2'])
            st.session_state.played_cards.add(row['Card3'])
    
    # Always remove Player A's fixed cards
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)

def save_rounds():
    """Saves round data to CSV."""
    st.session_state.rounds.to_csv(LOG_FILE, index=False)

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
    
    def find_patterns(df, patterns_to_watch):
    """  <-- ADDED TRIPLE QUOTES AND INDENTED
    Detects predefined sequences (patterns) in the outcomes of a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with an 'Outcome' column.
        patterns_to_watch (dict): A dictionary where keys are pattern names
                                  and values are lists of outcomes, e.g.,
                                  {'OOO_U': ['Over 21', 'Over 21', 'Over 21', 'Under 21']}
    Returns:
        dict: Counts of how many times each pattern was found.
    """  <-- ADDED TRIPLE QUOTES AND INDENTED

    pattern_counts = {name: 0 for name in patterns_to_watch.keys()}
    outcomes = df['Outcome'].tolist()

    for pattern_name, pattern_sequence in patterns_to_watch.items():
        pattern_len = len(pattern_sequence)
        for i in range(len(outcomes) - pattern_len + 1):
            if outcomes[i:i+pattern_len] == pattern_sequence:
                pattern_counts[pattern_name] += 1
    return pattern_counts

def reset_deck():
    """Resets the deck, starts a new Deck_ID."""
    st.session_state.current_deck_id += 1
    st.session_state.played_cards = set()
    # Always remove Player A's fixed cards for the new deck
    for card in PLAYER_A_FIXED_CARDS_STR:
        st.session_state.played_cards.add(card)
    st.success(f"New deck started! Current Deck ID: {st.session_state.current_deck_id}")

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


## Prediction Module

st.header("Next Round Prediction")

if not st.session_state.rounds.empty:
    current_deck_rounds_for_pred = st.session_state.rounds[st.session_state.rounds['Deck_ID'] == st.session_state.current_deck_id]

    if len(current_deck_rounds_for_pred) >= PREDICTION_ROUNDS_CONSIDERED:
        # Simple frequency-based prediction from recent rounds
        recent_outcomes = current_deck_rounds_for_pred['Outcome'].tail(PREDICTION_ROUNDS_CONSIDERED)
        over_freq = (recent_outcomes == 'Over 21').sum()
        under_freq = (recent_outcomes == 'Under 21').sum()
        exactly_freq = (recent_outcomes == 'Exactly 21').sum()

        total_recent_outcomes = over_freq + under_freq + exactly_freq

        if total_recent_outcomes > 0:
            if over_freq > under_freq and over_freq > exactly_freq:
                st.success(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} rounds, the next round might be **Over 21**.")
            elif under_freq > over_freq and under_freq > exactly_freq:
                st.info(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} rounds, the next round might be **Under 21**.")
            else:
                st.warning(f"Based on the last {PREDICTION_ROUNDS_CONSIDERED} rounds, the outcomes are balanced or 'Exactly 21' is common.")
        else:
            st.write(f"Not enough 'Over 21' or 'Under 21' outcomes in the last {PREDICTION_ROUNDS_CONSIDERED} rounds for a confident prediction.")
    else:
        st.write(f"Need at least {PREDICTION_ROUNDS_CONSIDERED} rounds in the current deck to start making predictions.")

    st.subheader("Remaining Cards Analysis (for current deck)")
    remaining_cards_count = len(ALL_CARDS) - len(st.session_state.played_cards)

    if remaining_cards_count > 0:
        st.write(f"**Cards remaining in current deck:** {remaining_cards_count}")
        
        # Count remaining high/low cards (simple analysis for prediction)
        remaining_high_cards = [card for card in available_cards_for_selection if card_values[card] >= 10]
        remaining_low_cards = [card for card in available_cards_for_selection if card_values[card] <= 5]

        st.write(f"- High value cards (10-K) remaining: {len(remaining_high_cards)}")
        st.write(f"- Low value cards (A-5) remaining: {len(remaining_low_cards)}")

        if len(remaining_high_cards) / remaining_cards_count > 0.3: # Arbitrary threshold
            st.info("Likely to see higher sums given the remaining high cards.")
        elif len(remaining_low_cards) / remaining_cards_count > 0.3:
            st.info("Likely to see lower sums given the remaining low cards.")
        else:
            st.info("Remaining cards distribution seems balanced.")
    else:
        st.write("All cards for the current deck seem to have been played!")
        st.warning("Consider starting a new deck.")

else:
    st.write("Play some rounds first to enable predictions!")


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
