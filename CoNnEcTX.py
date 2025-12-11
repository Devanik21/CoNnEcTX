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

st.title("🧠 Pure RL Connect X Game")
st.markdown("""
A **two-player Connect X game** solved using **Pure Reinforcement Learning** with self-play training.

1. **Configure Game**: Set board dimensions and win condition (X)
2. **Train Agents**: Watch two RL agents learn through self-play
3. **Play or Test**: Challenge the trained agent or watch it play!

**No Minimax. No Tree Search. Just Pure RL Magic! 🚀**
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
        
        # Advanced features
        self.experience_buffer = deque(maxlen=50000)
        self.model = {}  # State-action model
        self.priority_queue = []
        self.in_queue = set()
        
        # Tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.invalid_moves = 0
        
        # State visit counts for curiosity
        self.visit_counts = {}
        self.curiosity_weight = 0.5
        self.curiosity_decay = 0.9995
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), self.init_q_value)

    def get_state_value(self, board):
        """Evaluate a board state using learned Q-values (Intuition)"""
        state_tuple = tuple(map(tuple, board))
        valid_cols = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid_cols:
            return 0
        # The value of a state is the best Q-value we can get from it
        return max([self.get_q_value(state_tuple, col) for col in valid_cols], default=0)

    def minimax(self, board, depth, maximizing, alpha, beta, game_rules):
        """The AGI Brain: Lookahead Search + RL Evaluation"""
        rows, cols, win_len = game_rules
        
        # Helper to check wins on simulation board
        def check_win_sim(b, p):
            for r in range(rows):
                for c in range(cols):
                    if b[r,c] == p:
                        if c + win_len <= cols and np.all(b[r, c:c+win_len] == p): return True
                        if r + win_len <= rows and np.all(b[r:r+win_len, c] == p): return True
                        if r + win_len <= rows and c + win_len <= cols and all(b[r+i, c+i] == p for i in range(win_len)): return True
                        if r - win_len >= -1 and c + win_len <= cols and all(b[r-i, c+i] == p for i in range(win_len)): return True
            return False

        opponent = 3 - self.player_id
        if check_win_sim(board, self.player_id): return 10000  # We won
        if check_win_sim(board, opponent): return -10000       # Opponent won
        
        valid_moves = [c for c in range(cols) if board[0, c] == 0]
        if not valid_moves: return 0 # Draw

        # Base Case: If depth reached, use RL Intuition
        if depth == 0:
            return self.get_state_value(board)

        if maximizing:
            max_eval = -float('inf')
            for col in valid_moves:
                temp_board = board.copy()
                # Simulate drop
                for r in range(rows-1, -1, -1):
                    if temp_board[r, col] == 0:
                        temp_board[r, col] = self.player_id
                        break
                
                eval_score = self.minimax(temp_board, depth-1, False, alpha, beta, game_rules)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for col in valid_moves:
                temp_board = board.copy()
                # Simulate drop (Opponent)
                for r in range(rows-1, -1, -1):
                    if temp_board[r, col] == 0:
                        temp_board[r, col] = opponent
                        break
                        
                eval_score = self.minimax(temp_board, depth-1, True, alpha, beta, game_rules)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def choose_action(self, state, valid_moves, training=True, game_obj=None, minimax_depth=0):
        """Selection: Uses Epsilon-Greedy for Training, Minimax+RL for Play"""
        if not valid_moves:
            return None
        
        # TRAINING: Use fast Q-Learning (Epsilon Greedy)
        if training:
            if random.random() < self.epsilon:
                return random.choice(valid_moves)
            q_values = [(move, self.get_q_value(state, move)) for move in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_moves = [move for move, q in q_values if q == max_q]
            return random.choice(best_moves)
        
        # PLAYING: Use AGI (Minimax)
        current_board = np.array(state) # Convert tuple back to numpy
        
        # If depth is 0, just use Greedy Q-Learning (Fast)
        if minimax_depth == 0:
            q_values = [(move, self.get_q_value(state, move)) for move in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1]
            return random.choice([move for move, q in q_values if q == max_q])

        # If depth > 0, use Tree Search
        best_score = -float('inf')
        best_move = random.choice(valid_moves)
        
        game_rules = (current_board.shape[0], current_board.shape[1], 4)
        if game_obj: game_rules = (game_obj.rows, game_obj.cols, game_obj.win_length)

        for col in valid_moves:
            temp_board = current_board.copy()
            # Simulate move
            for r in range(current_board.shape[0]-1, -1, -1):
                if temp_board[r, col] == 0:
                    temp_board[r, col] = self.player_id
                    break
            
            # Call Minimax (Next turn is minimizing/opponent)
            score = self.minimax(temp_board, minimax_depth-1, False, -float('inf'), float('inf'), game_rules)
            
            if score > best_score:
                best_score = score
                best_move = col
                
        return best_move
    
    def get_intrinsic_reward(self, state):
        """Curiosity-driven exploration bonus"""
        visit_count = self.visit_counts.get(state, 0) + 1
        self.visit_counts[state] = visit_count
        intrinsic = (self.curiosity_weight / np.sqrt(visit_count))
        return intrinsic
    
    def update_q_value(self, state, action, reward, next_state, next_valid_moves, done):
        """Q-learning update with intrinsic motivation"""
        intrinsic_reward = self.get_intrinsic_reward(state)
        total_reward = reward + intrinsic_reward
        current_q = self.get_q_value(state, action)
        
        if done:
            target = total_reward
        else:
            if next_valid_moves:
                max_next_q = max([self.get_q_value(next_state, a) for a in next_valid_moves])
            else:
                max_next_q = 0
            target = total_reward + self.gamma * max_next_q
        
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q
        self.model[(state, action)] = (next_state, total_reward, done)
        
        td_error = abs(target - current_q)
        if td_error > 0.1:
            self.prioritized_update(state, action, td_error)
        return td_error
    
    def prioritized_update(self, state, action, priority, max_queue_size=1000):
        if (state, action) not in self.in_queue:
            if len(self.priority_queue) >= max_queue_size:
                heappop(self.priority_queue)
            heappush(self.priority_queue, (-abs(priority), state, action))
            self.in_queue.add((state, action))
    
    def planning_step(self, n_steps=10):
        """Dyna-Q style planning"""
        for _ in range(min(n_steps, len(self.priority_queue))):
            if not self.priority_queue:
                break
            _, state, action = heappop(self.priority_queue)
            self.in_queue.discard((state, action))
            
            if (state, action) in self.model:
                next_state, reward, done = self.model[(state, action)]
                current_q = self.get_q_value(state, action)
                if done:
                    target = reward
                else:
                    max_next_q = max([self.get_q_value(next_state, a) for a in range(7)], default=0)
                    target = reward + self.gamma * max_next_q
                new_q = current_q + self.lr * (target - current_q)
                self.q_table[(state, action)] = new_q
    
    def experience_replay(self, batch_size=32):
        """Learn from past experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        batch = random.sample(self.experience_buffer, batch_size)
        for state, action, reward, next_state, done, next_valid_moves in batch:
            self.update_q_value(state, action, reward, next_state, next_valid_moves, done)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.curiosity_weight = max(0.01, self.curiosity_weight * self.curiosity_decay)
    
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
        
        # Agent chooses action
        action = agent.choose_action(state, valid_moves, training=True)
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
                    r = 100 if p == winner else -100
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
                st.toast("Agents loaded!", icon="🧠")
                st.rerun()

train_button = st.sidebar.button("🚀 Train Agents (Self-Play)", use_container_width=True, type="primary")

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
    st.info("👈 Configure and create a game using the sidebar to begin!")
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
    st.divider()
    st.subheader("🎮 Interactive Play")
    
    # Initialize play session state
    if 'play_game' not in st.session_state:
        st.session_state.play_game = None
        st.session_state.play_history = []
        st.session_state.play_step = 0

    # Helper function to update view without full rerun
    def update_play_view(board_ph, info_ph, prog_ph):
        if not st.session_state.play_history:
            return
            
        curr_step = st.session_state.play_step
        total_steps = len(st.session_state.play_history) - 1
        
        # Update Progress
        prog_ph.progress(curr_step / max(total_steps, 1))
        
        # Render Board
        with board_ph.container():
            temp_game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
            # Restore board state from history
            temp_game.board = st.session_state.play_history[curr_step][0].copy()
            
            # If it's the last step, restore game status (win/loss)
            if curr_step == total_steps and st.session_state.play_game:
                temp_game.game_over = st.session_state.play_game.game_over
                temp_game.winner = st.session_state.play_game.winner
            
            fig = temp_game.render()
            st.pyplot(fig)
            plt.close() # Important: Close plot to free memory
        
        # Render Info
        with info_ph.container():
            st.markdown(f"### Move {curr_step} / {total_steps}")
            if curr_step > 0:
                _, player, col = st.session_state.play_history[curr_step]
                p_color = "🔴 Red" if player == 1 else "🟡 Yellow"
                st.info(f"**{p_color}** dropped in **Column {col}**")
            else:
                st.markdown("*Game Start*")

    # Layout for Controls
    play_col1, play_col2, play_col3 = st.columns([1, 1, 1])
    
    with play_col1:
        # ADDED: Randomness Slider for Watching
        watch_randomness = st.slider("Randomness (Varied Games)", 0.0, 1.0, 0.1, 0.05, 
                                     help="If 0, agents always play the same 'perfect' game. Increase to see variations.")
        
        if st.button("🎬 New Game (Watch Agents)", use_container_width=True, type="primary"):
            # Start a completely new game
            test_game = ConnectXGame(config['rows'], config['cols'], config['win_length'])
            state = test_game.reset()
            
            # Play the entire game and record history
            # History format: (board_copy, player_who_just_moved, col_dropped)
            play_history = [(test_game.board.copy(), None, None)]
            move_count = 0
            
            while not test_game.game_over and move_count < max_moves:
                current_player = test_game.current_player
                agent = agent1 if current_player == 1 else agent2
                valid_moves = test_game.get_valid_moves()
                
                if not valid_moves:
                    break
                
                # --- UPDATE: Logic to add variety ---
                # If randomness > 0, occasionally pick a random valid move
                if random.random() < watch_randomness:
                    action = random.choice(valid_moves)
                else:
                    # Otherwise use the Agent's Brain
                    action = agent.choose_action(
                        state, 
                        valid_moves, 
                        training=False, 
                        game_obj=test_game, 
                        minimax_depth=minimax_depth
                    )
                # ------------------------------------
                
                state, reward, done, info = test_game.make_move(action)
                play_history.append((test_game.board.copy(), current_player, action))
                move_count += 1
            
            # Store in session state
            st.session_state.play_game = test_game
            st.session_state.play_history = play_history
            st.session_state.play_step = 0
            st.rerun()

    # Dynamic Placeholders
    progress_placeholder = st.empty()
    col_display1, col_display2 = st.columns([2, 1])
    board_placeholder = col_display1.empty()
    info_placeholder = col_display2.empty()

    # Navigation Buttons
    with play_col2:
        if st.session_state.play_game and st.session_state.play_step < len(st.session_state.play_history) - 1:
            if st.button("➡️ Next Move", use_container_width=True):
                st.session_state.play_step += 1
                # Update logic is handled below, no rerun needed usually, 
                # but Streamlit buttons trigger rerun anyway.
    
    with play_col3:
        if st.session_state.play_game and st.session_state.play_step > 0:
            if st.button("⬅️ Previous Move", use_container_width=True):
                st.session_state.play_step -= 1

    # Render the current state (Runs on every script execution)
    if st.session_state.play_game:
        update_play_view(board_placeholder, info_placeholder, progress_placeholder)
        
        # End of Game Message
        if st.session_state.play_step == len(st.session_state.play_history) - 1:
            if st.session_state.play_game.game_over:
                res_container = st.container()
                if st.session_state.play_game.winner:
                    winner = st.session_state.play_game.winner
                    color = "🔴 Red" if winner == 1 else "🟡 Yellow"
                    res_container.success(f"🏆 {color} Wins!")
                else:
                    res_container.info("🤝 It's a Draw!")
    else:
        board_placeholder.info("👆 Click 'New Game' to watch the agents play!")
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Game Board", use_container_width=True):
        st.session_state.game.reset()
        st.session_state.play_game = None
        st.session_state.play_history = []
        st.session_state.play_step = 0
        st.rerun()
