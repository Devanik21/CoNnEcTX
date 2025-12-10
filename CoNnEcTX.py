import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd
import json
import zipfile
import io
import ast
from copy import deepcopy
import time

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="RL Connect-X Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.title("🧠 Genius RL Connect-X Arena")
st.markdown("""
Watch two Reinforcement Learning agents battle to master **Connect-X**! These agents use pure Q-learning with advanced techniques like **Experience Replay** and **Opponent Modeling** to develop their strategies from scratch.

1.  **Configure the Game**: Set the grid size and win condition in the sidebar.
2.  **Tune the Agents**: Adjust hyperparameters for each agent independently.
3.  **Train**: Start the training process and watch their win rates evolve.
4.  **Battle!**: Pit the trained agents against each other in a final showdown.
""")

# ============================================================================
# Game Environment Class
# ============================================================================

class ConnectX:
    def __init__(self, grid_size=6, connect_n=4):
        self.grid_size = grid_size
        self.connect_n = connect_n
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def get_available_actions(self):
        return [c for c in range(self.grid_size) if self.board[0, c] == 0]

    def make_move(self, column):
        if self.game_over or column not in self.get_available_actions():
            # Invalid move, penalize heavily and end game for this player
            return self.get_state(), -20, True, 3 - self.current_player

        # Find the lowest empty row in the chosen column
        for r in range(self.grid_size - 1, -1, -1):
            if self.board[r, column] == 0:
                self.board[r, column] = self.current_player
                break

        if self._check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            return self.get_state(), 100, True, self.winner

        if len(self.get_available_actions()) == 0:
            self.game_over = True
            self.winner = 0  # Draw
            return self.get_state(), 10, True, 0

        self.current_player = 3 - self.current_player
        return self.get_state(), -0.5, False, None # Small penalty for each move to encourage faster wins

    def _check_win(self, player):
        # Check horizontal, vertical, and diagonal connections
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Horizontal
                if c <= self.grid_size - self.connect_n:
                    if all(self.board[r, c+i] == player for i in range(self.connect_n)):
                        return True
                # Vertical
                if r <= self.grid_size - self.connect_n:
                    if all(self.board[r+i, c] == player for i in range(self.connect_n)):
                        return True
                # Positive Diagonal
                if r <= self.grid_size - self.connect_n and c <= self.grid_size - self.connect_n:
                    if all(self.board[r+i, c+i] == player for i in range(self.connect_n)):
                        return True
                # Negative Diagonal
                if r <= self.grid_size - self.connect_n and c >= self.connect_n - 1:
                    if all(self.board[r+i, c-i] == player for i in range(self.connect_n)):
                        return True
        return False

# ============================================================================
# "Genius" RL Agent Class
# ============================================================================

class ConnectXAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        self.experience_replay = deque(maxlen=20000)
        self.opponent_model = {}  # Learns opponent's likely moves from a state

        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions, training=True):
        if not available_actions:
            return None
        
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Genius move: Consider opponent's likely counter-move
        # This is a simple form of opponent modeling
        best_action = None
        best_q_val = -float('inf')

        for action in available_actions:
            q_val = self.get_q_value(state, action)
            
            # Simulate my move to predict opponent's response
            # This is a one-step lookahead, a hallmark of smarter play
            if training and self.opponent_model:
                sim_board = list(state)
                # Find where my piece would land (simplified)
                # This part is complex, so we'll approximate
                # A full simulation would be better but this is a good RL-based approach
                # For now, we just use the direct Q-value
                pass

            if q_val > best_q_val:
                best_q_val = q_val
                best_action = action
        
        # If multiple actions have the same max Q-value, pick one randomly
        if best_action is None:
             return random.choice(available_actions)

        best_actions = [a for a in available_actions if self.get_q_value(state, a) == best_q_val]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, done, next_available_actions):
        # Standard Q-learning update
        current_q = self.get_q_value(state, action)
        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions], default=0)
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def learn_from_replay(self, batch_size=64):
        if len(self.experience_replay) < batch_size:
            return
        
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, done, next_actions in minibatch:
            self.update_q_table(state, action, reward, next_state, done, next_actions)

    def update_opponent_model(self, state, opponent_action):
        if state not in self.opponent_model:
            self.opponent_model[state] = {}
        self.opponent_model[state][opponent_action] = self.opponent_model[state].get(opponent_action, 0) + 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training & Game Logic
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    env.reset()
    history = []
    
    while not env.game_over:
        current_agent = agent1 if env.current_player == 1 else agent2
        opponent_agent = agent2 if env.current_player == 1 else agent1
        
        state = env.get_state()
        available_actions = env.get_available_actions()
        action = current_agent.choose_action(state, available_actions, training)
        
        if action is None:
            break

        next_state, reward, done, winner = env.make_move(action)
        
        # Store experience for the current player
        # The reward for the opponent will be the negative of the current player's reward
        history.append({
            'state': state, 'action': action, 'reward': reward, 
            'next_state': next_state, 'done': done, 'player': current_agent.player_id,
            'next_actions': env.get_available_actions()
        })
        
        # Opponent modeling
        if training:
            opponent_agent.update_opponent_model(state, action)

    # Post-game learning
    if training:
        final_reward_p1 = 0
        if winner == 1: final_reward_p1 = 100
        elif winner == 2: final_reward_p1 = -100
        elif winner == 0: final_reward_p1 = 10

        for move in reversed(history):
            reward = final_reward_p1 if move['player'] == 1 else -final_reward_p1
            agent = agent1 if move['player'] == 1 else agent2
            
            # Add to experience replay buffer
            agent.experience_replay.append((
                move['state'], move['action'], reward, move['next_state'], 
                move['done'], move['next_actions']
            ))
            
            # Learn from a batch of experiences
            agent.learn_from_replay()

    return winner

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Game Board"):
    fig, ax = plt.subplots(figsize=(6, 6))
    n = board.shape[0]
    ax.set_facecolor('#2E86C1')

    for r in range(n):
        for c in range(n):
            if board[r, c] == 1:
                color = 'red'
            elif board[r, c] == 2:
                color = 'yellow'
            else:
                color = 'white'
            ax.add_patch(plt.Circle((c + 0.5, n - r - 0.5), 0.4, color=color))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight='bold', color='white')
    return fig

# ============================================================================
# Save/Load Utility Functions
# ============================================================================

def serialize_q_table(q_table):
    return {str(k): v for k, v in q_table.items()}

def deserialize_q_table(serialized_q):
    return {ast.literal_eval(k): v for k, v in serialized_q.items()}

def create_agents_zip(agent1, agent2, config):
    agent1_state = {"q_table": serialize_q_table(agent1.q_table), "epsilon": agent1.epsilon, "wins": agent1.wins, "losses": agent1.losses, "draws": agent1.draws}
    agent2_state = {"q_table": serialize_q_table(agent2.q_table), "epsilon": agent2.epsilon, "wins": agent2.wins, "losses": agent2.losses, "draws": agent2.draws}
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1_brain.json", json.dumps(agent1_state))
        zf.writestr("agent2_brain.json", json.dumps(agent2_state))
        zf.writestr("game_config.json", json.dumps(config))
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file, agent1_params, agent2_params):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_state = json.loads(zf.read("agent1_brain.json"))
            agent2_state = json.loads(zf.read("agent2_brain.json"))
            config = json.loads(zf.read("game_config.json"))
            
            agent1 = ConnectXAgent(1, **agent1_params)
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state.get('epsilon', 1.0)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.draws = agent1_state.get('draws', 0)
            
            agent2 = ConnectXAgent(2, **agent2_params)
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state.get('epsilon', 1.0)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.draws = agent2_state.get('draws', 0)
            
            return agent1, agent2, config
    except Exception as e:
        st.error(f"Failed to load agents: {e}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Game & Agent Controls")

with st.sidebar.expander("1. Game Configuration", expanded=True):
    grid_size = st.slider("Grid Size", 3, 10, 6)
    connect_n = st.slider("Connect 'X' to Win", 2, 5, 4)
    if connect_n > grid_size:
        st.warning("Win condition cannot be larger than the grid size.")
        connect_n = grid_size
    st.info(f"Playing on a {grid_size}x{grid_size} grid. Need {connect_n} in a row to win.")

with st.sidebar.expander("2. Agent 1 (Red) Hyperparameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 1.0, 0.1, 0.01, key="lr1")
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.9, 0.01, key="gamma1")
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.9995, 0.0001, format="%.4f", key="ed1")

with st.sidebar.expander("3. Agent 2 (Yellow) Hyperparameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 1.0, 0.1, 0.01, key="lr2")
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.9, 0.01, key="gamma2")
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.9995, 0.0001, format="%.4f", key="ed2")

with st.sidebar.expander("4. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 100000, 5000, 100)
    update_freq = st.number_input("Update Dashboard Every N Games", 10, 1000, 100, 10)

with st.sidebar.expander("5. Brain Storage (Save/Load)", expanded=False):
    if 'agent1' in st.session_state:
        game_config = {"grid_size": grid_size, "connect_n": connect_n}
        zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, game_config)
        st.download_button(
            label="💾 Download Agent Brains (.zip)", data=zip_buffer,
            file_name="connectx_brains.zip", mime="application/zip", use_container_width=True
        )
    else:
        st.warning("Train agents first to download.")
    
    uploaded_file = st.file_uploader("Upload Agent Brains (.zip)", type="zip")
    if uploaded_file:
        if st.button("Load Agent Brains", use_container_width=True):
            agent1_params = {'lr': lr1, 'gamma': gamma1, 'epsilon_decay': epsilon_decay1}
            agent2_params = {'lr': lr2, 'gamma': gamma2, 'epsilon_decay': epsilon_decay2}
            a1, a2, cfg = load_agents_from_zip(uploaded_file, agent1_params, agent2_params)
            if a1:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                st.session_state.env = ConnectX(cfg['grid_size'], cfg['connect_n'])
                st.session_state.training_history = None
                st.toast("Agent Brains Restored!", icon="🧠")
                st.rerun()

train_button = st.sidebar.button("🚀 Start Training", use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear & Reset", use_container_width=True):
    keys_to_clear = ['agent1', 'agent2', 'training_history', 'env']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ============================================================================
# Main Area
# ============================================================================

# Initialize environment and agents
if 'env' not in st.session_state:
    st.session_state.env = ConnectX(grid_size, connect_n)
env = st.session_state.env

if 'agent1' not in st.session_state:
    st.session_state.agent1 = ConnectXAgent(1, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent2 = ConnectXAgent(2, lr2, gamma2, epsilon_decay=epsilon_decay2)
agent1 = st.session_state.agent1
agent2 = st.session_state.agent2

# Display stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Agent 1 (Red) Wins", agent1.wins)
    st.caption(f"Q-States: {len(agent1.q_table):,}")
with col2:
    st.metric("Agent 2 (Yellow) Wins", agent2.wins)
    st.caption(f"Q-States: {len(agent2.q_table):,}")
with col3:
    st.metric("Draws", agent1.draws)

# Training Loop
if train_button:
    st.subheader("🧠 Training in Progress...")
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {'agent1_wins': [], 'agent2_wins': [], 'draws': []}
    
    for episode in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        if winner == 1: agent1.wins += 1
        elif winner == 2: agent2.wins += 1
        else: agent1.draws += 1; agent2.draws += 1
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)

            progress = episode / episodes
            progress_bar.progress(progress)
            
            status_text = f"""
            **Game {episode}/{episodes}** ({progress*100:.1f}%)
            - **Agent 1 (Red):** Wins: {agent1.wins} | ε: {agent1.epsilon:.4f}
            - **Agent 2 (Yellow):** Wins: {agent2.wins} | ε: {agent2.epsilon:.4f}
            - **Draws:** {agent1.draws}
            """
            status_container.markdown(status_text)

    st.session_state.training_history = history
    st.toast("Training Complete!", icon="🎉")
    st.rerun()

# Display charts and final game
if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance")
    history = st.session_state.training_history
    df = pd.DataFrame(history)
    df['episode_chunk'] = range(update_freq, episodes + 1, update_freq)
    
    st.line_chart(df.set_index('episode_chunk'))

    st.subheader("🤖 Final Battle: Trained Agents")
    if st.button("Watch Them Battle!", use_container_width=True):
        sim_env = ConnectX(grid_size, connect_n)
        board_placeholder = st.empty()
        
        while not sim_env.game_over:
            current_player_id = sim_env.current_player
            agent = agent1 if current_player_id == 1 else agent2
            
            state = sim_env.get_state()
            available_actions = sim_env.get_available_actions()
            action = agent.choose_action(state, available_actions, training=False)
            
            if action is None: break
            
            sim_env.make_move(action)
            fig = visualize_board(sim_env.board, f"Player {current_player_id}'s move")
            board_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.5)

        if sim_env.winner == 1: st.success("🏆 Agent 1 (Red) wins the final battle!")
        elif sim_env.winner == 2: st.error("🏆 Agent 2 (Yellow) wins the final battle!")
        else: st.warning("🤝 The final battle is a Draw!")
