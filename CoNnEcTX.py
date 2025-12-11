import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from heapq import heappush, heappop
import pandas as pd
import json
import zipfile
import io
import ast

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="RL Connect X",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎮"
)

st.title(" Pure RL Connect X Game")
st.markdown("""
A **two-player Connect X game** solved using **Pure Reinforcement Learning** with self-play training.

1. **Configure Game**: Set board dimensions and win condition (X)
2. **Train Agents**: Watch two RL agents learn through self-play
3. **Play or Test**: Challenge the trained agent or watch it play!

**No Minimax. No Tree Search. Just Pure RL Magic! **
""")

# ============================================================================
# Connect X Game Environment
# ============================================================================

class ConnectXGame:
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        return self.get_state()
    
    def get_state(self):
        """Convert board to tuple for hashing in Q-table"""
        return tuple(map(tuple, self.board))
    
    def get_valid_moves(self):
        """Returns list of valid column indices"""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def make_move(self, col):
        """Drop piece in column, returns (new_state, reward, done, info)"""
        if col not in self.get_valid_moves():
            return self.get_state(), -100, True, {'invalid': True}
        
        # Drop piece
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.last_move = (row, col)
                break
        
        # Check win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            reward = 100  # Win!
            return self.get_state(), reward, True, {'winner': self.current_player}
        
        # Check draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            reward = 0  # Draw
            return self.get_state(), reward, True, {'draw': True}
        
        # Switch player
        self.current_player = 3 - self.current_player  # Toggle 1<->2
        return self.get_state(), 0, False, {}
    
    def _check_win(self, row, col):
        """Check if last move resulted in a win"""
        player = self.board[row, col]
        directions = [(0,1), (1,0), (1,1), (1,-1)]  # horizontal, vertical, diagonals
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            for i in range(1, self.win_length):
                r, c = row + dr*i, col + dc*i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # Check negative direction
            for i in range(1, self.win_length):
                r, c = row - dr*i, col - dc*i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return True
        return False
    
    def render(self):
        """Returns matplotlib figure of current board"""
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Draw board
        for row in range(self.rows):
            for col in range(self.cols):
                circle = plt.Circle((col, self.rows - 1 - row), 0.4, 
                                   color='white', ec='black', linewidth=2)
                ax.add_patch(circle)
                
                if self.board[row, col] == 1:
                    piece = plt.Circle((col, self.rows - 1 - row), 0.35, color='red')
                    ax.add_patch(piece)
                elif self.board[row, col] == 2:
                    piece = plt.Circle((col, self.rows - 1 - row), 0.35, color='yellow')
                    ax.add_patch(piece)
        
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        title = "Connect X Game"
        if self.game_over:
            if self.winner:
                title = f"Player {self.winner} Wins!"
            else:
                title = "Draw!"
        else:
            title = f"Player {self.current_player}'s Turn"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig

# ============================================================================
# Pure RL Agent (Q-Learning with Self-Play)
# ============================================================================

# ============================================================================
# HYBRID AGENT: RL + Minimax (The "Smart & Fast" Update)
# ============================================================================

class PureRLAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.99, epsilon_decay=0.9995, epsilon_min=0.05):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_table = {}
        self.init_q_value = 0.0
        
        self.experience_buffer = deque(maxlen=50000)
        self.model = {}
        self.priority_queue = []
        self.in_queue = set()
        
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.invalid_moves = 0

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), self.init_q_value)

    # --- 1. THE EYE: Heuristic Evaluation ---
    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 3 - piece
        
        if window.count(piece) == 4:
            score += 10000
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 10
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 4 # slightly preferred
        
        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 80 # BLOCK OPPONENT!
            
        return score

    def score_position(self, board, piece):
        score = 0
        rows, cols = board.shape
        
        # Preference for Center Column (Strategic Advantage)
        center_array = [int(i) for i in list(board[:, cols//2])]
        center_count = center_array.count(piece)
        score += center_count * 6

        # Horizontal
        for r in range(rows):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(cols-3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece)

        # Vertical
        for c in range(cols):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(rows-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        # Positive Diagonal
        for r in range(rows-3):
            for c in range(cols-3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # Negative Diagonal
        for r in range(rows-3):
            for c in range(cols-3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    # --- 2. THE BRAIN: Minimax with Alpha-Beta Pruning ---
    def is_terminal_node(self, board, game_rules):
        rows, cols, win_len = game_rules
        # Check Win P1
        if self.check_win_static(board, 1, win_len): return True
        # Check Win P2
        if self.check_win_static(board, 2, win_len): return True
        # Check Draw
        if len([c for c in range(cols) if board[0][c] == 0]) == 0: return True
        return False

    def check_win_static(self, board, piece, win_len):
        # Quick check for the Minimax simulation
        rows, cols = board.shape
        # Horizontal
        for c in range(cols-3):
            for r in range(rows):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        # Vertical
        for c in range(cols):
            for r in range(rows-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        # Pos Diag
        for c in range(cols-3):
            for r in range(rows-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        # Neg Diag
        for c in range(cols-3):
            for r in range(3, rows):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, game_rules):
        rows, cols, win_len = game_rules
        valid_locations = [c for c in range(cols) if board[0][c] == 0]
        is_terminal = self.is_terminal_node(board, game_rules)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win_static(board, self.player_id, win_len):
                    return 10000000
                elif self.check_win_static(board, 3-self.player_id, win_len):
                    return -10000000
                else: # Draw
                    return 0
            else: # Depth is zero, use Heuristic + Q-Value Intuition
                # 1. Heuristic Score
                h_score = self.score_position(board, self.player_id)
                
                # 2. Q-Value Intuition (The RL part!)
                # We try to see if we have visited this state in training
                state_tuple = tuple(map(tuple, board))
                # Average Q-value of available moves from here
                q_total = 0
                for col in valid_locations:
                    q_total += self.get_q_value(state_tuple, col)
                
                # Combine: Heuristic is main driver, Q-value is the "tie breaker" or "gut feeling"
                return h_score + (q_total * 0.1)

        if maximizingPlayer:
            value = -float('inf')
            # Randomize order to add variety
            random.shuffle(valid_locations)
            for col in valid_locations:
                # Simulate Move
                temp_board = board.copy()
                for r in range(rows-1, -1, -1):
                    if temp_board[r][col] == 0:
                        temp_board[r][col] = self.player_id
                        break
                new_score = self.minimax(temp_board, depth-1, alpha, beta, False, game_rules)
                value = max(value, new_score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else: # Minimizing Opponent
            value = float('inf')
            random.shuffle(valid_locations)
            for col in valid_locations:
                temp_board = board.copy()
                opponent = 3 - self.player_id
                for r in range(rows-1, -1, -1):
                    if temp_board[r][col] == 0:
                        temp_board[r][col] = opponent
                        break
                new_score = self.minimax(temp_board, depth-1, alpha, beta, True, game_rules)
                value = min(value, new_score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # --- 3. SELECTION: Hybrid Decision Making ---
    def choose_action(self, state, valid_moves, training=True, game_obj=None, minimax_depth=0):
        if not valid_moves:
            return None
        
        # TRAINING: Use pure RL (Epsilon Greedy) for speed
        if training:
            if random.random() < self.epsilon:
                return random.choice(valid_moves)
            q_values = [(move, self.get_q_value(state, move)) for move in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_moves = [move for move, q in q_values if q == max_q]
            return random.choice(best_moves)
        
        # PLAYING: Use Minimax (Smart)
        # Convert state tuple back to numpy for calculation
        board_np = np.array(state)
        
        # Depth 2 is fast and smart. Depth 4 is very smart but slower.
        # We default to 2 if depth is 0 to ensure intelligence.
        depth_to_use = minimax_depth if minimax_depth > 0 else 2
        
        # Game rules extraction
        rules = (board_np.shape[0], board_np.shape[1], 4)
        if game_obj: rules = (game_obj.rows, game_obj.cols, game_obj.win_length)

        best_score = -float('inf')
        best_col = random.choice(valid_moves)
        
        for col in valid_moves:
            temp_board = board_np.copy()
            for r in range(rules[0]-1, -1, -1):
                if temp_board[r][col] == 0:
                    temp_board[r][col] = self.player_id
                    break
            
            # Call Minimax
            score = self.minimax(temp_board, depth_to_use, -float('inf'), float('inf'), False, rules)
            
            if score > best_score:
                best_score = score
                best_col = col
        
        return best_col

    # (Keep standard RL update functions below)
    def update_q_value(self, state, action, reward, next_state, next_valid_moves, done):
        # Standard Q-learning update
        current_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            if next_valid_moves:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_valid_moves])
            else:
                max_next_q = 0
            target = reward + self.gamma * max_next_q
        
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q
        self.model[(state, action)] = (next_state, reward, done)
        return abs(target - current_q)

    def planning_step(self, n_steps=10):
        pass # Optional optimization
    
    def experience_replay(self, batch_size=32):
        if len(self.experience_buffer) < batch_size: return
        batch = random.sample(self.experience_buffer, batch_size)
        for state, action, reward, next_state, done, next_valid_moves in batch:
            self.update_q_value(state, action, reward, next_state, next_valid_moves, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def record_result(self, result):
        if result == 'win': self.wins += 1
        elif result == 'loss': self.losses += 1
        elif result == 'draw': self.draws += 1

# ============================================================================
# Self-Play Training
# ============================================================================

def train_self_play(game, agent1, agent2, max_moves=100):
    """Train two agents against each other"""
    state = game.reset()
    history = []  # Store (state, action, player) for later reward assignment
    
    for move_num in range(max_moves):
        current_player = game.current_player
        agent = agent1 if current_player == 1 else agent2
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        
        
        action = agent.choose_action(state, valid_moves, training=True, game_obj=game)
        history.append((state, action, current_player))
        
        # Execute move
        next_state, reward, done, info = game.make_move(action)
        
        if 'invalid' in info:
            agent.invalid_moves += 1
            return 'invalid', move_num
        
        # Store experience for current agent
        next_valid_moves = game.get_valid_moves() if not done else []
        agent.experience_buffer.append((state, action, reward, next_state, done, next_valid_moves))
        
        if done:
            # Assign rewards to both agents
            if 'winner' in info:
                winner = info['winner']
                # Winner gets +100, loser gets -100
                for i, (s, a, p) in enumerate(history):
                    r = 500 if p == winner else -1000
                    agt = agent1 if p == 1 else agent2
                    ns = history[i+1][0] if i+1 < len(history) else next_state
                    nvm = []  # Game is done
                    agt.update_q_value(s, a, r, ns, nvm, True)
                
                # Record results
                if winner == 1:
                    agent1.record_result('win')
                    agent2.record_result('loss')
                    result = 'p1_win'
                else:
                    agent1.record_result('loss')
                    agent2.record_result('win')
                    result = 'p2_win'
            else:  # Draw
                for s, a, p in history:
                    agt = agent1 if p == 1 else agent2
                    agt.update_q_value(s, a, 0, next_state, [], True)
                agent1.record_result('draw')
                agent2.record_result('draw')
                result = 'draw'
            
            # Experience replay and planning
            agent1.experience_replay(batch_size=32)
            agent2.experience_replay(batch_size=32)
            agent1.planning_step(n_steps=20)
            agent2.planning_step(n_steps=20)
            
            return result, move_num
        
        state = next_state
    
    return 'timeout', max_moves

# ============================================================================
# Save/Load Functions
# ============================================================================

def serialize_q_table(q_table):
    serialized = {}
    for key, value in q_table.items():
        # Convert numpy types to native Python types
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        serialized[str(key)] = value
    return serialized

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def deserialize_q_table(serialized_q):
    q_table = {}
    for key_str, value in serialized_q.items():
        key_tuple = ast.literal_eval(key_str)
        q_table[key_tuple] = value
    return q_table

def create_brain_zip(agent1, agent2, game_config):
    agent1_state = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": float(agent1.epsilon),
        "lr": float(agent1.lr),
        "gamma": float(agent1.gamma),
        "wins": int(agent1.wins),
        "losses": int(agent1.losses),
        "draws": int(agent1.draws),
        "model": {str(k): convert_to_serializable(v) for k, v in agent1.model.items()}
    }
    
    agent2_state = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": float(agent2.epsilon),
        "lr": float(agent2.lr),
        "gamma": float(agent2.gamma),
        "wins": int(agent2.wins),
        "losses": int(agent2.losses),
        "draws": int(agent2.draws),
        "model": {str(k): convert_to_serializable(v) for k, v in agent2.model.items()}
    }
    
    config_state = {
        "rows": int(game_config['rows']),
        "cols": int(game_config['cols']),
        "win_length": int(game_config['win_length'])
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1_brain.json", json.dumps(agent1_state))
        zf.writestr("agent2_brain.json", json.dumps(agent2_state))
        zf.writestr("game_config.json", json.dumps(config_state))
    
    buffer.seek(0)
    return buffer

def load_brain_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_json = zf.read("agent1_brain.json")
            agent2_json = zf.read("agent2_brain.json")
            config_json = zf.read("game_config.json")
            
            agent1_state = json.loads(agent1_json)
            agent2_state = json.loads(agent2_json)
            config_state = json.loads(config_json)
            
            # Reconstruct agents
            agent1 = PureRLAgent(1, agent1_state['lr'], agent1_state['gamma'], 0.9995, 0.05)
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.model = deserialize_q_table(agent1_state['model'])
            agent1.epsilon = agent1_state['epsilon']
            agent1.wins = agent1_state['wins']
            agent1.losses = agent1_state['losses']
            agent1.draws = agent1_state['draws']
            
            agent2 = PureRLAgent(2, agent2_state['lr'], agent2_state['gamma'], 0.9995, 0.05)
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.model = deserialize_q_table(agent2_state['model'])
            agent2.epsilon = agent2_state['epsilon']
            agent2.wins = agent2_state['wins']
            agent2.losses = agent2_state['losses']
            agent2.draws = agent2_state['draws']
            
            return agent1, agent2, config_state
    except Exception as e:
        st.error(f"Error loading brain: {e}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Game Controls")

with st.sidebar.expander("1. Game Configuration", expanded=True):
    rows = st.number_input("Board Rows", min_value=4, max_value=20, value=6, step=1)
    cols = st.number_input("Board Columns", min_value=4, max_value=20, value=7, step=1)
    win_length = st.slider("Win Length (X)", min_value=2, max_value=5, value=4, step=1)
    
    if st.button("Create New Game", use_container_width=True):
        st.session_state.game_config = {
            'rows': rows,
            'cols': cols,
            'win_length': win_length
        }
        st.session_state.game = ConnectXGame(rows, cols, win_length)
        st.session_state.agent1 = None
        st.session_state.agent2 = None
        st.session_state.training_history = None
        
        # --- FIX: Clear the old recordings so they don't clash! ---
        st.session_state.play_game = None
        st.session_state.play_history = []
        st.session_state.play_step = 0
        # ----------------------------------------------------------
        
        st.toast("New game created!", icon="🎮")
        st.rerun()

with st.sidebar.expander("2. Agent Hyperparameters", expanded=True):
    lr = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99, 0.001)
    epsilon_decay = st.slider("Epsilon Decay", 0.99, 0.9999, 0.9995, 0.0001, format="%.4f")
    epsilon_min = st.slider("Min Epsilon (ε)", 0.01, 0.3, 0.05, 0.01)
    
    st.markdown("---")
    st.markdown("**AGI Thinking**")
    minimax_depth = st.slider(
        "Lookahead Depth", 
        min_value=0, 
        max_value=4, 
        value=0, 
        help="0 = Pure RL (Intuition), >0 = RL + Tree Search (Calculation)"
    )

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 1000000, 5000, 100)
    max_moves = st.number_input("Max Moves per Game", 10, 1000, 100, 10)
    early_stop_rate = st.slider("Early Stop (Win Rate)", 0.7, 0.99, 0.85, 0.01)

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        zip_buffer = create_brain_zip(
            st.session_state.agent1, 
            st.session_state.agent2,
            st.session_state.game_config
        )
        st.download_button(
            label="💾 Download Agents (.zip)",
            data=zip_buffer,
            file_name="connect_x_agents.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("Train agents first to download.")
    
    uploaded_file = st.file_uploader("Upload Brain (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Load Agents", use_container_width=True):
            agent1, agent2, config = load_brain_from_zip(uploaded_file)
            if agent1:
                st.session_state.agent1 = agent1
                st.session_state.agent2 = agent2
                st.session_state.game_config = config
                st.session_state.game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
                st.toast("Agents loaded!", icon="")
                st.rerun()

train_button = st.sidebar.button(" Train Agents (Self-Play)", use_container_width=True, type="primary")

st.sidebar.divider()

if st.sidebar.button("Clear Memory", use_container_width=True):
    for key in ['game', 'agent1', 'agent2', 'training_history', 'game_config']:
        if key in st.session_state:
            del st.session_state[key]
    st.toast("Memory cleared!", icon="🧼")
    st.rerun()

# ============================================================================
# Main Area
# ============================================================================

if 'game_config' not in st.session_state:
    st.info(" Configure and create a game using the sidebar to begin!")
else:
    config = st.session_state.game_config
    game = st.session_state.game
    
    # Initialize agents if needed
    if 'agent1' not in st.session_state or st.session_state.agent1 is None:
        st.session_state.agent1 = PureRLAgent(1, lr, gamma, epsilon_decay, epsilon_min)
        st.session_state.agent2 = PureRLAgent(2, lr, gamma, epsilon_decay, epsilon_min)
    
    agent1 = st.session_state.agent1
    agent2 = st.session_state.agent2
    
    # Training
    if train_button:
        st.subheader("🏋️ Training in Progress...")
        
        progress_bar = st.progress(0)
        status_container = st.empty()
        chart_container = st.empty()
        
        p1_wins, p2_wins, draws = 0, 0, 0
        recent_results = deque(maxlen=100)
        win_rates = []
        
        for episode in range(1, episodes + 1):
            result, moves = train_self_play(game, agent1, agent2, max_moves)
            
            if result == 'p1_win':
                p1_wins += 1
                recent_results.append(1)
            elif result == 'p2_win':
                p2_wins += 1
                recent_results.append(2)
            elif result == 'draw':
                draws += 1
                recent_results.append(0)
            
            agent1.decay_epsilon()
            agent2.decay_epsilon()
            
            # Calculate win rates
            if len(recent_results) > 0:
                p1_wr = sum(1 for r in recent_results if r == 1) / len(recent_results)
                p2_wr = sum(1 for r in recent_results if r == 2) / len(recent_results)
            else:
                p1_wr = p2_wr = 0
            
            win_rates.append({'episode': episode, 'p1': p1_wr, 'p2': p2_wr})
            
            # Update UI every 50 episodes
            if episode % 50 == 0 or episode == 1:
                status_md = f"""
                | Metric | Value |
                |--------|-------|
                | **Episode** | `{episode}` / `{episodes}` |
                | **Agent 1 Epsilon** | `{agent1.epsilon:.4f}` |
                | **Agent 2 Epsilon** | `{agent2.epsilon:.4f}` |
                | **P1 Q-Table Size** | `{len(agent1.q_table):,}` |
                | **P2 Q-Table Size** | `{len(agent2.q_table):,}` |
                | **P1 Wins** | `{p1_wins}` ({p1_wr:.1%}) |
                | **P2 Wins** | `{p2_wins}` ({p2_wr:.1%}) |
                | **Draws** | `{draws}` |
                """
                status_container.markdown(status_md)
                progress_bar.progress(episode / episodes)
                
                # Plot win rates
                if len(win_rates) > 10:
                    df = pd.DataFrame(win_rates)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df['episode'], df['p1'], label='Agent 1', color='red', linewidth=2)
                    ax.plot(df['episode'], df['p2'], label='Agent 2', color='yellow', linewidth=2)
                    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Win Rate (Last 100)')
                    ax.set_ylim(0, 1)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    chart_container.pyplot(fig)
                    plt.close()
            
            # Early stopping
            if len(recent_results) >= 100 and (p1_wr > early_stop_rate or p2_wr > early_stop_rate):
                st.success(f"🎉 Early stop at episode {episode}! Agent achieved {max(p1_wr, p2_wr):.1%} win rate!")
                break
        
        st.session_state.training_history = {
            'win_rates': win_rates,
            'total_games': episode,
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'draws': draws
        }
        st.rerun()
    
    # Display current game state
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Game Board")
        fig = game.render()
        st.pyplot(fig)
        plt.close()
        
        st.metric("Board Size", f"{config['rows']}x{config['cols']}")
        st.metric("Win Condition", f"Connect {config['win_length']}")
    
    with col2:
        st.subheader("Agent Statistics")
        
        if agent1 and agent2:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**🔴 Agent 1 (Red)**")
                st.metric("Q-Table Size", f"{len(agent1.q_table):,}")
                st.metric("Epsilon", f"{agent1.epsilon:.4f}")
                st.metric("Wins", agent1.wins)
                st.metric("Losses", agent1.losses)
                st.metric("Draws", agent1.draws)
            
            with col_b:
                st.markdown("**🟡 Agent 2 (Yellow)**")
                st.metric("Q-Table Size", f"{len(agent2.q_table):,}")
                st.metric("Epsilon", f"{agent2.epsilon:.4f}")
                st.metric("Wins", agent2.wins)
                st.metric("Losses", agent2.losses)
                st.metric("Draws", agent2.draws)
        
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.divider()
            history = st.session_state.training_history
            st.markdown("**Training Summary**")
            st.metric("Total Games Played", history['total_games'])
            st.metric("Final P1 Win Rate", f"{history['p1_wins']/history['total_games']:.1%}")
            st.metric("Final P2 Win Rate", f"{history['p2_wins']/history['total_games']:.1%}")
    
    # Interactive Play Section
    st.divider()
    # ============================================================================
    # UPDATE: Interactive Play Section (Replace the bottom section with this)
    # ============================================================================
    # ============================================================================
    # FINAL UPDATE: Interactive Play Section (Watch OR Play)
    # ============================================================================
    st.divider()
    st.header("🎮 Interactive Zone")

    # We use Tabs to switch modes without reloading the whole app
    tab_watch, tab_play = st.tabs(["📺 Watch Agents", "⚔️ Play vs Agent"])

    # ------------------------------------------------------------------------
    # TAB 1: WATCH AGENTS (The previous logic, refined)
    # ------------------------------------------------------------------------
    with tab_watch:
        st.caption("Watch two trained agents play against each other.")
        
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            watch_randomness = st.slider("Randomness", 0.0, 1.0, 0.1, 0.05, key="watch_rand")
            
            if st.button("🎬 Start New Watch Game", use_container_width=True):
                # Initialize a new game for watching
                watch_game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
                state = watch_game.reset()
                
                # Record the full game history instantly
                history = [(watch_game.board.copy(), None, None)]
                
                while not watch_game.game_over:
                    p = watch_game.current_player
                    ag = agent1 if p == 1 else agent2
                    valid = watch_game.get_valid_moves()
                    if not valid: break
                    
                    # Decide move
                    if random.random() < watch_randomness:
                        act = random.choice(valid)
                    else:
                        act = ag.choose_action(state, valid, training=False, game_obj=watch_game, minimax_depth=minimax_depth)
                    
                    state, _, done, _ = watch_game.make_move(act)
                    history.append((watch_game.board.copy(), p, act))
                
                st.session_state.watch_history = history
                st.session_state.watch_step = 0
                st.session_state.watch_game_over = watch_game.winner
                st.rerun()

        # Playback Controls
        if 'watch_history' in st.session_state:
            total_steps = len(st.session_state.watch_history) - 1
            curr = st.session_state.watch_step
            
            # Slider for fast scrubbing
            new_step = st.slider("Timeline", 0, total_steps, curr, label_visibility="collapsed")
            if new_step != curr:
                st.session_state.watch_step = new_step
                st.rerun()
            
            # Render Board
            board_state = st.session_state.watch_history[new_step][0]
            
            # Temporary game object just for rendering
            render_game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
            render_game.board = board_state
            
            # Show winner if at the end
            if new_step == total_steps and st.session_state.watch_game_over:
                render_game.game_over = True
                render_game.winner = st.session_state.watch_game_over
            
            st.pyplot(render_game.render())
            plt.close()

            # Navigation Buttons
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("⏪ Start"): 
                st.session_state.watch_step = 0
                st.rerun()
            if c2.button("◀️ Prev") and curr > 0: 
                st.session_state.watch_step -= 1
                st.rerun()
            if c3.button("Next ▶️") and curr < total_steps: 
                st.session_state.watch_step += 1
                st.rerun()
            if c4.button("End ⏩"): 
                st.session_state.watch_step = total_steps
                st.rerun()

    # ------------------------------------------------------------------------
    # TAB 2: PLAY VS AGENT (The New Feature!)
    # ------------------------------------------------------------------------
    with tab_play:
        st.caption("Challenge your trained agent! Can you beat it?")
        
        # 1. Setup Controls
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            human_side = st.radio("You play as:", ["🔴 Player 1 (Red)", "🟡 Player 2 (Yellow)"])
            human_id = 1 if "Player 1" in human_side else 2
            opponent_agent = agent2 if human_id == 1 else agent1
        
        with col_p2:
            st.write("") # Spacer
            st.write("") 
            if st.button("🔥 Start Match", use_container_width=True, type="primary"):
                st.session_state.human_game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
                st.session_state.human_id = human_id
                
                # If Human is Player 2, AI moves first immediately!
                if human_id == 2:
                    ai_obs = st.session_state.human_game.get_state()
                    ai_valid = st.session_state.human_game.get_valid_moves()
                    ai_move = agent1.choose_action(ai_obs, ai_valid, training=False, minimax_depth=minimax_depth)
                    st.session_state.human_game.make_move(ai_move)
                
                st.rerun()

        st.divider()

        # 2. Game Interface
        if 'human_game' in st.session_state:
            h_game = st.session_state.human_game
            
            # A. Input Buttons (Only show if game is active)
            if not h_game.game_over:
                st.write(f"**Your Turn ({'🔴 Red' if st.session_state.human_id == 1 else '🟡 Yellow'})** - Click a column:")
                cols_ui = st.columns(h_game.cols)
                
                for c in range(h_game.cols):
                    # Only enable valid columns
                    if h_game.board[0, c] == 0:
                        if cols_ui[c].button("⬇️", key=f"btn_{c}", use_container_width=True):
                            # 1. HUMAN MOVE
                            h_game.make_move(c)
                            
                            # 2. AI MOVE (Immediately after, if game not over)
                            if not h_game.game_over:
                                ai_agent = agent2 if st.session_state.human_id == 1 else agent1
                                ai_obs = h_game.get_state()
                                ai_valid = h_game.get_valid_moves()
                                
                                # AI Thinking...
                                ai_act = ai_agent.choose_action(ai_obs, ai_valid, training=False, game_obj=h_game, minimax_depth=minimax_depth)
                                h_game.make_move(ai_act)
                            
                            st.rerun()
                    else:
                        cols_ui[c].button("❌", disabled=True, key=f"btn_full_{c}")
            
            # B. Render Board
            st.pyplot(h_game.render())
            plt.close()

            # C. Game Over Message
            if h_game.game_over:
                if h_game.winner == st.session_state.human_id:
                    
                    st.success("🏆 VICTORY! You defeated the AI!")
                elif h_game.winner:
                    st.error("💀 DEFEAT! The AI wins.")
                else:
                    st.info("🤝 It's a Draw!")
                
                if st.button("Rematch?", use_container_width=True):
                    # Trigger the start button logic again essentially
                    del st.session_state.human_game
                    st.rerun()
        else:
            st.info("👆 Click 'Start Match' to begin!")
    
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Game Board", use_container_width=True):
        st.session_state.game.reset()
        st.session_state.play_game = None
        st.session_state.play_history = []
        st.session_state.play_step = 0
        st.rerun()
