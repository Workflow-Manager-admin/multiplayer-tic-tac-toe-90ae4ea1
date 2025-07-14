-- Tic Tac Toe Game Database Schema

-- Table for user accounts
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for sessions (simple session info)
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Table for individual tic tac toe games
CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    player_x_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    player_o_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    game_state VARCHAR(32) NOT NULL, -- e.g., 'in_progress', 'finished', etc
    winner INTEGER REFERENCES users(id),
    current_turn CHAR(1) NOT NULL, -- 'X' or 'O'
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Move history: all moves made in games
CREATE TABLE moves (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK(position >= 0 AND position < 9), -- 0-8 for board index
    move_number INTEGER NOT NULL, -- order in game (from 1 to 9)
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_move_per_number UNIQUE (game_id, move_number),
    CONSTRAINT unique_position_per_game UNIQUE (game_id, position)
);

-- Useful indices
CREATE INDEX idx_moves_game_id ON moves (game_id);
CREATE INDEX idx_games_player_x_id ON games (player_x_id);
CREATE INDEX idx_games_player_o_id ON games (player_o_id);
CREATE INDEX idx_sessions_user_id ON sessions (user_id);
