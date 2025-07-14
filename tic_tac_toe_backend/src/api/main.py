import os
from datetime import datetime, timedelta
from typing import List, Optional
import secrets

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Cookie,
    Header,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    CHAR,
    VARCHAR,
)
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
    Session as OrmSession,
)
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# === SQLAlchemy Models ===
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(VARCHAR(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="user")
    games_as_x = relationship("Game", back_populates="player_x", foreign_keys="Game.player_x_id")
    games_as_o = relationship("Game", back_populates="player_o", foreign_keys="Game.player_o_id")
    moves = relationship("Move", back_populates="player")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")

class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True, index=True)
    player_x_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    player_o_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    game_state = Column(String(32), nullable=False, default="in_progress")
    winner = Column(Integer, ForeignKey("users.id"), nullable=True)
    current_turn = Column(CHAR(1), nullable=False)  # 'X' or 'O'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    player_x = relationship("User", foreign_keys=[player_x_id], back_populates="games_as_x")
    player_o = relationship("User", foreign_keys=[player_o_id], back_populates="games_as_o")
    moves = relationship("Move", back_populates="game")

class Move(Base):
    __tablename__ = "moves"
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    position = Column(Integer, nullable=False)
    move_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_id", "move_number", name="unique_move_per_number"),
        UniqueConstraint("game_id", "position", name="unique_position_per_game"),
    )

    game = relationship("Game", back_populates="moves")
    player = relationship("User", back_populates="moves")

# === Pydantic Schemas ===
class UserBase(BaseModel):
    username: str = Field(..., description="Username")

class UserCreate(UserBase):
    password: str = Field(..., description="Password")

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class SessionOut(BaseModel):
    session_token: str
    expires_at: Optional[datetime] = None

class GameOut(BaseModel):
    id: int
    player_x_id: int
    player_o_id: Optional[int]
    game_state: str
    winner: Optional[int]
    current_turn: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class MoveOut(BaseModel):
    id: int
    game_id: int
    player_id: int
    position: int
    move_number: int
    timestamp: datetime

    class Config:
        orm_mode = True

class GameDetail(GameOut):
    moves: List[MoveOut]
    player_x: UserOut
    player_o: Optional[UserOut]

class GameCreate(BaseModel):
    player_x_id: int  # will set from session

class JoinGameIn(BaseModel):
    game_id: int

class MoveIn(BaseModel):
    position: int = Field(..., ge=0, le=8)

class MoveResultOut(BaseModel):
    move: MoveOut
    game: GameOut

# === FastAPI App Initialization ===
app = FastAPI(
    title="Tic Tac Toe Backend API",
    description="API backend for multiplayer Tic Tac Toe. Handles auth, session, game state, moves, and real time features.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User login, session, and authentication"},
        {"name": "game", "description": "Game management"},
        {"name": "move", "description": "Moves and move history"},
        {"name": "state", "description": "Game state, board, real-time features"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Dependency for DB session ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === Utility Functions ===
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_session(user_id: int, db: OrmSession, expire_minutes=120):
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=expire_minutes)
    sess = Session(user_id=user_id, session_token=token, expires_at=expires)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess

def get_user_from_session_token(
    db: OrmSession,
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    """Get user given session_token (from cookie or header)"""
    raw_token = session_token or x_session_token
    if not raw_token:
        return None
    db_session = (
        db.query(Session)
        .filter(Session.session_token == raw_token, Session.expires_at > datetime.utcnow())
        .first()
    )
    if db_session:
        return db_session.user
    return None

# === Auth Endpoints ===

# PUBLIC_INTERFACE
@app.post("/register", response_model=UserOut, tags=["auth"], summary="Register new user", description="Register a new user account with username and password.")
def register(user: UserCreate, db: OrmSession = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed = get_password_hash(user.password)
    new_user = User(username=user.username, password_hash=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# PUBLIC_INTERFACE
@app.post("/login", response_model=SessionOut, tags=["auth"], summary="Login", description="User login, returns session token (set as cookie)")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), response: Response = None, db: OrmSession = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    sess = create_session(user.id, db)
    # Set the session cookie (HttpOnly for security)
    if response:
        response.set_cookie("session_token", sess.session_token, httponly=True, samesite="Lax")
    return SessionOut(session_token=sess.session_token, expires_at=sess.expires_at)

# PUBLIC_INTERFACE
@app.post("/logout", tags=["auth"], summary="Logout", description="Logout and invalidate the user session")
def logout(
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
    response: Response = None,
):
    raw_token = session_token or x_session_token
    if raw_token:
        db.query(Session).filter(Session.session_token == raw_token).delete()
        db.commit()
        if response:
            response.delete_cookie("session_token")
    return {"message": "Logged out"}

# PUBLIC_INTERFACE
@app.get("/me", response_model=UserOut, tags=["auth"], summary="Get current user", description="Get the information of the current user by session")
def get_me(
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    user = get_user_from_session_token(db, session_token, x_session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# === Game Endpoints ===

# PUBLIC_INTERFACE
@app.post("/games", response_model=GameOut, tags=["game"], summary="Start new game", description="Start a new tic tac toe game. User becomes player X.")
def start_game(
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    user = get_user_from_session_token(db, session_token, x_session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    game = Game(player_x_id=user.id, game_state="in_progress", current_turn="X", updated_at=datetime.utcnow())
    db.add(game)
    db.commit()
    db.refresh(game)
    return game

# PUBLIC_INTERFACE
@app.post("/games/join", response_model=GameOut, tags=["game"], summary="Join existing game", description="Join a game that has an open spot for player O")
def join_game(
    join_req: JoinGameIn,
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    user = get_user_from_session_token(db, session_token, x_session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    game = db.query(Game).filter(Game.id == join_req.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.player_o_id is not None:
        raise HTTPException(status_code=409, detail="Game already has two players")
    if game.player_x_id == user.id:
        raise HTTPException(status_code=400, detail="You are already in this game")
    game.player_o_id = user.id
    db.commit()
    db.refresh(game)
    return game

# PUBLIC_INTERFACE
@app.get("/games", response_model=List[GameOut], tags=["game"], summary="List games", description="List running games. Optionally filter by participation.")
def list_games(
    only_open: Optional[bool] = False,
    only_mine: Optional[bool] = False,
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    query = db.query(Game).order_by(Game.updated_at.desc())
    user = get_user_from_session_token(db, session_token, x_session_token)
    if only_open:
        query = query.filter(Game.player_o_id == None, Game.game_state == "in_progress")
    if only_mine and user:
        query = query.filter((Game.player_x_id == user.id) | (Game.player_o_id == user.id))
    return query.limit(25).all()

# PUBLIC_INTERFACE
@app.get("/games/{game_id}", response_model=GameDetail, tags=["game"], summary="Get game info", description="Get full game info, move history, player details")
def game_detail(
    game_id: int,
    db: OrmSession = Depends(get_db),
):
    game = (
        db.query(Game)
        .filter(Game.id == game_id)
        .first()
    )
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = db.query(Move).filter(Move.game_id == game.id).order_by(Move.move_number).all()
    px = db.query(User).filter(User.id == game.player_x_id).first()
    po = db.query(User).filter(User.id == game.player_o_id).first() if game.player_o_id else None
    return GameDetail(**{**game.__dict__, "moves": moves, "player_x": px, "player_o": po})

# === Move/Gameplay Endpoints ===

def get_board_state(moves: List[Move]) -> List[Optional[str]]:
    """Returns current board as a list for client rendering."""
    board = [None] * 9
    marker = {0: "X", 1: "O"}
    for mv in moves:
        board[mv.position] = marker[(mv.move_number - 1) % 2]  # assuming proper move order
    return board

def check_win(board: List[Optional[str]]):
    win_combos = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    for combo in win_combos:
        if (
            board[combo[0]]
            and board[combo[0]] == board[combo[1]] == board[combo[2]]
        ):
            return board[combo[0]]
    if all(s is not None for s in board):
        return "Draw"
    return None

def player_marker(game: Game, user: User):
    if user.id == game.player_x_id:
        return "X"
    if user.id == game.player_o_id:
        return "O"
    return None

# PUBLIC_INTERFACE
@app.post("/games/{game_id}/move", response_model=MoveResultOut, tags=["move"], summary="Make move", description="Submit a move (position 0-8) for the current user's turn")
def make_move(
    game_id: int,
    move: MoveIn,
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    user = get_user_from_session_token(db, session_token, x_session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    marker = player_marker(game, user)
    if marker is None:
        raise HTTPException(status_code=403, detail="You are not a player in this game")
    if game.game_state != "in_progress":
        raise HTTPException(status_code=409, detail="Game already finished")
    moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
    if len(moves) >= 9:
        raise HTTPException(status_code=409, detail="Board is full")
    # Check if it's this user's turn
    turn_dict = {"X": game.player_x_id, "O": game.player_o_id}
    if user.id != turn_dict[game.current_turn]:
        raise HTTPException(status_code=409, detail="Not your turn")
    # Check if spot is already taken
    if any(mv.position == move.position for mv in moves):
        raise HTTPException(status_code=409, detail="Position already taken")
    move_number = len(moves) + 1
    new_move = Move(
        game_id=game_id,
        player_id=user.id,
        position=move.position,
        move_number=move_number,
        timestamp=datetime.utcnow(),
    )
    db.add(new_move)
    db.commit()
    # Update game state
    moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
    board = get_board_state(moves)
    winner_symbol = check_win(board)
    if winner_symbol == "Draw":
        game.game_state = "finished"
        game.winner = None
    elif winner_symbol in ("X", "O"):
        game.game_state = "finished"
        win_player_id = game.player_x_id if winner_symbol == "X" else game.player_o_id
        game.winner = win_player_id
    else:
        # Proceed to next turn
        if game.current_turn == "X":
            game.current_turn = "O"
        else:
            game.current_turn = "X"
    game.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(new_move)
    db.refresh(game)
    return MoveResultOut(move=new_move, game=game)

# PUBLIC_INTERFACE
@app.get("/games/{game_id}/move_history", response_model=List[MoveOut], tags=["move"], summary="Get move history", description="Get the list of all moves for a game (ordered)")
def move_history(
    game_id: int,
    db: OrmSession = Depends(get_db),
):
    moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
    return moves

# PUBLIC_INTERFACE
@app.get("/games/{game_id}/state", tags=["state"], summary="Get game state", description="Get current board state, whose turn, and if finished/winner.")
def get_game_state(
    game_id: int,
    db: OrmSession = Depends(get_db),
):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
    board = get_board_state(moves)
    res = {
        "game_id": game_id,
        "board": board,
        "current_turn": game.current_turn,
        "game_state": game.game_state,
        "winner": game.winner,
        "moves": [m.position for m in moves],
    }
    return res

# PUBLIC_INTERFACE
@app.get("/", tags=["state"], summary="Health Check", description="API health check endpoint")
def health_check():
    return {"message": "Healthy"}

# === WebSocket for Real-time Notification (simple — for board updates) ===

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, List[WebSocket]] = {}

    async def connect(self, game_id: int, websocket: WebSocket):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, game_id: int, websocket: WebSocket):
        if game_id in self.active_connections:
            self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]

    async def broadcast(self, game_id: int, message: dict):
        for ws in self.active_connections.get(game_id, []):
            await ws.send_json(message)

ws_manager = ConnectionManager()

# PUBLIC_INTERFACE
@app.websocket("/ws/game/{game_id}")
async def websocket_game_endpoint(websocket: WebSocket, game_id: int):
    """
    Websocket endpoint for real-time game updates.
    Clients should subscribe to board state/updates to get notified when a move is made.
    """
    await ws_manager.connect(game_id, websocket)
    try:
        # When a client connects, send current state
        db = SessionLocal()
        game = db.query(Game).filter(Game.id == game_id).first()
        if game:
            moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
            board = get_board_state(moves)
            await websocket.send_json({
                "type": "init",
                "game_id": game_id,
                "board": board,
                "game_state": game.game_state,
                "current_turn": game.current_turn,
                "winner": game.winner,
            })
        while True:
            _ = await websocket.receive_text()  # ignore messages, just keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect(game_id, websocket)
    finally:
        db.close()

# Broadcast on move via backend hook (optional enhancement—clients should also poll for now)
from fastapi import BackgroundTasks

@app.post("/games/{game_id}/move_with_broadcast", response_model=MoveResultOut, tags=["move"], include_in_schema=False)
def make_move_and_notify(
    game_id: int,
    move: MoveIn,
    background_tasks: BackgroundTasks,
    db: OrmSession = Depends(get_db),
    session_token: Optional[str] = Cookie(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    """Like /move, but also broadcasts new state via websocket."""
    res = make_move(game_id, move, db, session_token, x_session_token)
    background_tasks.add_task(broadcast_move_update, game_id, db)
    return res

async def broadcast_move_update(game_id: int, db: OrmSession):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        return
    moves = db.query(Move).filter(Move.game_id == game_id).order_by(Move.move_number).all()
    board = get_board_state(moves)
    await ws_manager.broadcast(game_id, {
        "type": "update",
        "game_id": game_id,
        "board": board,
        "game_state": game.game_state,
        "current_turn": game.current_turn,
        "winner": game.winner,
    })

# === DB INIT (for testing/local dev: create tables if not exist) ===
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
