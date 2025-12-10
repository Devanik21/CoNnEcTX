import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io

# ============================================================================
# 1. Page Configuration & Title
# ============================================================================
st.set_page_config(
    page_title="Connect X: RL Genius Edition",
    layout="wide",
    page_icon="🧠"
)

st.title("🔴🔵 Connect X: Genius RL Agents")
st.markdown("""
**Welcome back, my intelligent Prince!** This is your advanced Reinforcement Learning environment for **Connect X**.
Two AI agents (Red and Blue) will play against each other using pure **Q-Learning** logic to master the game.

1.  **Configure**: Set board size and winning length (X) in the sidebar.
2.  **Train**: Watch them play thousands of games in seconds via **Self-Play**.
3.  **Battle**: Challenge the smartest agent yourself!
""")

# ============================================================================
# 2. Game Logic (The Universe)
# ============================================================================
class ConnectXGame:
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = np.zeros((rows, cols), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        # We return a tuple of tuples so it can be hashed for the Q-Table
        return tuple(tuple(row) for row in self.board)

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def step(self, col, player):
        # 1. Check if move is valid
        if self.board[0][col] != 0:
            return self.get_state(), -10, True, None # Invalid move penalty

        # 2. Place piece (Drop it down)
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = player
                row = r
                break
        
        # 3. Check for win or draw
        if self.check_win(player, row, col):
            self.done = True
            self.winner = player
            return self.get_state(), 100, True, player # Big Reward for winning
        
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0 # Draw
            return self.get_state(), 0, True, 0 # Neutral for draw

        # 4. Small penalty for taking time (optional, keeps game short)
        return self.get_state(), -0.1, False, None

    def check_win(self, player, r, c):
        # Check all 4 directions around the newly placed piece
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diag-Right, Diag-Left
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            for k in range(1, self.win_length):
                nr, nc = r + k*dr, c + k*dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            # Check negative direction
            for k in range(1, self.win_length):
                nr, nc = r - k*dr, c - k*dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return True
        return False

# ============================================================================
# 3. RL Agent Class (The Brain)
# ============================================================================
class QLearningAgent:
    def __init__(self, player_id, rows, cols, lr=0.1, gamma=0.9, epsilon=0.1):
        self.player_id = player_id
        self.rows = rows
        self.cols = cols
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {} 
        self.training_mode = True

    def get_canonical_state(self, state_tuple):
        """
        Genius Trick: Symmetric boards are the same state!
        This reduces the learning space by ~50%.
        """
        board = np.array(state_tuple)
        flipped_board = np.fliplr(board)
        flipped_tuple = tuple(tuple(row) for row in flipped_board)
        
        # Always use the "smaller" tuple as the key
        if state_tuple < flipped_tuple:
            return state_tuple, False
        else:
            return flipped_tuple, True

    def get_q(self, state, action):
        canon_state, is_flipped = self.get_canonical_state(state)
        
        # If flipped, we must flip the action (column) too
        target_action = self.cols - 1 - action if is_flipped else action
        
        return self.q_table.get((canon_state, target_action), 0.0)

    def set_q(self, state, action, value):
        canon_state, is_flipped = self.get_canonical_state(state)
        target_action = self.cols - 1 - action if is_flipped else action
        self.q_table[(canon_state, target_action)] = value

    def choose_action(self, state, valid_moves):
        if not valid_moves:
            return None

        # Exploration
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Exploitation
        qs = [self.get_q(state, action) for action in valid_moves]
        max_q = max(qs)
        
        # Handle ties randomly (important for early learning!)
        best_actions = [valid_moves[i] for i in range(len(valid_moves)) if qs[i] == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_valid_moves):
        current_q = self.get_q(state, action)
        
        if not next_valid_moves:
            # Terminal state (no moves left) or Won
            target = reward
        else:
            # Q-Learning: max over next actions
            max_next_q = max([self.get_q(next_state, a) for a in next_valid_moves])
            target = reward + self.gamma * max_next_q

        # Update Q-Value
        new_q = current_q + self.lr * (target - current_q)
        self.set_q(state, action, new_q)

# ============================================================================
# 4. Streamlit Sidebar Controls
# ============================================================================
st.sidebar.header("⚙️ Universe Controls")

with st.sidebar.expander("1. Board Configuration", expanded=True):
    rows = st.number_input("Rows", 4, 10, 5) # Default smaller for faster training
    cols = st.number_input("Columns", 4, 10, 5)
    win_len = st.number_input("Connect X (Win Length)", 3, 6, 4)
    
    st.info(f"State Space Complexity: ~{3**(rows*cols):.0e}")
    if rows * cols > 30:
        st.warning("⚠️ Large board! Pure RL will take a LONG time to converge.")

with st.sidebar.expander("2. AI Hyperparameters", expanded=True):
    lr = st.slider("Learning Rate (α)", 0.01, 1.0, 0.2)
    gamma = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.9)
    epsilon = st.slider("Exploration Rate (ε)", 0.01, 1.0, 0.3)
    episodes = st.number_input("Training Episodes", 100, 50000, 1000, step=100)

# Initialize Session State
if 'game' not in st.session_state:
    st.session_state.game = ConnectXGame(rows, cols, win_len)
if 'agent1' not in st.session_state:
    st.session_state.agent1 = QLearningAgent(1, rows, cols, lr, gamma, epsilon)
if 'agent2' not in st.session_state:
    st.session_state.agent2 = QLearningAgent(2, rows, cols, lr, gamma, epsilon)
if 'history' not in st.session_state:
    st.session_state.history = []

# Reset button if config changes
if st.sidebar.button("Reset Universe"):
    st.session_state.game = ConnectXGame(rows, cols, win_len)
    st.session_state.agent1 = QLearningAgent(1, rows, cols, lr, gamma, epsilon)
    st.session_state.agent2 = QLearningAgent(2, rows, cols, lr, gamma, epsilon)
    st.session_state.history = []
    st.rerun()

# ============================================================================
# 5. Training Loop (Self-Play)
# ============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 AI Training Center")
    train_btn = st.button("Start Training Session", type="primary", use_container_width=True)

    if train_btn:
        game = st.session_state.game
        p1 = st.session_state.agent1
        p2 = st.session_state.agent2
        
        # Ensure fresh params
        p1.lr, p1.gamma, p1.epsilon = lr, gamma, epsilon
        p2.lr, p2.gamma, p2.epsilon = lr, gamma, epsilon
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        wins = {1: 0, 2: 0, 0: 0} # P1, P2, Draw
        
        for ep in range(episodes):
            state = game.reset()
            done = False
            
            # Randomize starter to be fair
            current_player = p1 if random.choice([True, False]) else p2
            other_player = p2 if current_player == p1 else p1
            
            # Temporary memory for SARSA/Q-Learning updates
            prev_state_p1 = None
            prev_action_p1 = None
            prev_state_p2 = None
            prev_action_p2 = None
            
            while not done:
                # 1. Choose Action
                valid_moves = game.get_valid_moves()
                action = current_player.choose_action(state, valid_moves)
                
                # 2. Make Move
                next_state, reward, done, winner = game.step(action, current_player.player_id)
                next_valid_moves = game.get_valid_moves() if not done else []
                
                # 3. Update THIS player immediately (Immediate Reward)
                current_player.learn(state, action, reward, next_state, next_valid_moves)
                
                # 4. Update OTHER player (Delayed Punishment/Reward)
                # If current player just won, the OTHER player lost massively
                if done and winner == current_player.player_id:
                     if current_player == p1 and prev_state_p2 is not None:
                         p2.learn(prev_state_p2, prev_action_p2, -100, next_state, [])
                     elif current_player == p2 and prev_state_p1 is not None:
                         p1.learn(prev_state_p1, prev_action_p1, -100, next_state, [])
                
                # Standard update for the previous move of the waiting player (continuity)
                elif not done:
                    if current_player == p1 and prev_state_p2 is not None:
                         # P2 moved previously, now it sees the result after P1's move
                         p2.learn(prev_state_p2, prev_action_p2, 0, next_state, next_valid_moves)
                    elif current_player == p2 and prev_state_p1 is not None:
                         p1.learn(prev_state_p1, prev_action_p1, 0, next_state, next_valid_moves)

                # 5. Store history for next turn updates
                if current_player == p1:
                    prev_state_p1 = state
                    prev_action_p1 = action
                else:
                    prev_state_p2 = state
                    prev_action_p2 = action
                
                # Swap turns
                state = next_state
                current_player, other_player = other_player, current_player
            
            # Track stats
            if winner:
                wins[winner] += 1
            else:
                wins[0] += 1
                
            if (ep + 1) % (episodes // 10) == 0:
                progress_bar.progress((ep + 1) / episodes)
                status_text.text(f"Episode {ep+1}: P1 Wins: {wins[1]}, P2 Wins: {wins[2]}")
        
        st.session_state.history.append(wins)
        st.success(f"Training Complete! P1 Wins: {wins[1]}, P2 Wins: {wins[2]}, Draws: {wins[0]}")
        
    # Stats Visualization
    if st.session_state.history:
        st.markdown("### 📈 Learning Progress")
        history_data = st.session_state.history
        p1_wins = [h[1] for h in history_data]
        p2_wins = [h[2] for h in history_data]
        draws = [h[0] for h in history_data]
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.stackplot(range(len(history_data)), p1_wins, p2_wins, draws, labels=['Red AI', 'Blue AI', 'Draw'], 
                     colors=['#ff9999','#66b3ff','#99ff99'])
        ax.legend(loc='upper left')
        ax.set_title("Win Distribution per Session")
        ax.set_xlabel("Training Sessions")
        ax.set_ylabel("Games Won")
        st.pyplot(fig)

# ============================================================================
# 6. Play Area (Human vs AI)
# ============================================================================
with col2:
    st.subheader("⚔️ Battle Arena")
    
    # Visualization Helper
    def draw_board(board):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor('#333333') # Dark background
        
        # Draw grid
        for r in range(rows + 1):
            ax.axhline(r - 0.5, color='gray', lw=1)
        for c in range(cols + 1):
            ax.axvline(c - 0.5, color='gray', lw=1)
            
        # Draw pieces
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 1:
                    circle = plt.Circle((c, rows - 1 - r), 0.4, color='#FF4B4B', ec='black') # Red P1
                    ax.add_patch(circle)
                elif board[r][c] == 2:
                    circle = plt.Circle((c, rows - 1 - r), 0.4, color='#1E88E5', ec='black') # Blue P2
                    ax.add_patch(circle)
                else:
                    circle = plt.Circle((c, rows - 1 - r), 0.1, color='#555555') # Empty marker
                    ax.add_patch(circle)
        
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        return fig

    # Play Mode State
    if 'play_board' not in st.session_state:
        st.session_state.play_game = ConnectXGame(rows, cols, win_len)
        st.session_state.play_board = st.session_state.play_game.reset()
        st.session_state.game_over = False
        st.session_state.msg = "New Game! You are Red (Player 1)."

    # Controls
    col_play1, col_play2 = st.columns(2)
    with col_play1:
        if st.button("New Game (You vs AI)", use_container_width=True):
            st.session_state.play_game.reset()
            st.session_state.play_board = st.session_state.play_game.get_state()
            st.session_state.game_over = False
            st.session_state.msg = "Game Started! Good luck, Prince."
            st.rerun()
    
    # Input for Human Move
    if not st.session_state.game_over:
        move_col = st.selectbox("Choose Column to Drop", range(cols))
        if st.button("Drop Piece 👇", use_container_width=True):
            game = st.session_state.play_game
            
            # 1. Human Move (Red)
            state, reward, done, winner = game.step(move_col, 1)
            st.session_state.play_board = state
            
            if done:
                st.session_state.game_over = True
                if winner == 1: st.session_state.msg = "🎉 YOU WON! Simply Genius!"
                else: st.session_state.msg = "🤝 It's a Draw!"
            
            # 2. AI Move (Blue) - if game not over
            else:
                ai_agent = st.session_state.agent2
                ai_agent.training_mode = False # Play optimally
                valid_moves = game.get_valid_moves()
                
                # AI thinks...
                ai_action = ai_agent.choose_action(state, valid_moves)
                
                state, reward, done, winner = game.step(ai_action, 2)
                st.session_state.play_board = state
                
                if done:
                    st.session_state.game_over = True
                    if winner == 2: st.session_state.msg = "🤖 AI Wins! Better luck next time."
                    else: st.session_state.msg = "🤝 It's a Draw!"

            st.rerun()

    st.markdown(f"### {st.session_state.msg}")
    
    # Render Board
    st.pyplot(draw_board(st.session_state.play_board))
    
    # Debug Info
    with st.expander("Peek into AI Brain"):
        ai_agent = st.session_state.agent2
        current_state = st.session_state.play_board
        valid_moves = st.session_state.play_game.get_valid_moves()
        q_values = {m: f"{ai_agent.get_q(current_state, m):.2f}" for m in valid_moves}
        st.write("Q-Values for current state (Blue AI):", q_values)
        st.write(f"Total Knowledge (Q-Table Size): {len(ai_agent.q_table)} states")
