import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import io, random, heapq, time

# Puzzle logic
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def index_to_rc(idx):
    return divmod(idx, 3)  # (row, col)

def rc_to_index(r,c):
    return r*3 + c

def neighbors(state):
    """Return list of neighbor states reachable by moving the blank."""
    zero_idx = state.index(0)
    r, c = index_to_rc(zero_idx)
    moves = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nidx = rc_to_index(nr,nc)
            ns = list(state)
            ns[zero_idx], ns[nidx] = ns[nidx], ns[zero_idx]
            moves.append(tuple(ns))
    return moves

def manhattan(s, goal=GOAL):
    dist = 0
    # s is tuple length 9; tiles 1..8
    for idx, tile in enumerate(s):
        if tile == 0: continue
        goal_idx = goal.index(tile)
        r1,c1 = index_to_rc(idx)
        r2,c2 = index_to_rc(goal_idx)
        dist += abs(r1-r2) + abs(c1-c2)
    return dist

def is_solvable(state):
    """Check inversions ignoring blank for 8-puzzle."""
    flat = [x for x in state if x != 0]
    inv = sum(1 for i in range(len(flat)) for j in range(i+1,len(flat)) if flat[i] > flat[j])
    return inv % 2 == 0

def astar(start, goal=GOAL, max_nodes=200000):
    """Return list of states from start to goal (inclusive) or None if not found."""
    start = tuple(start)
    if start == goal:
        return [start]
    open_heap = []
    g = {start: 0}
    fstart = manhattan(start, goal)
    heapq.heappush(open_heap, (fstart, 0, start))
    came_from = {}
    closed = set()
    counter = 1
    nodes = 0
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        nodes += 1
        if nodes > max_nodes:
            return None  # fail gracefully for very large search
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        closed.add(current)
        for nb in neighbors(list(current)):
            if nb in closed:
                continue
            tentative_g = g[current] + 1
            if nb not in g or tentative_g < g[nb]:
                came_from[nb] = current
                g[nb] = tentative_g
                f = tentative_g + manhattan(nb, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))
    return None

# Initialize session_state
if "state" not in st.session_state:
    st.session_state.state = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # solved puzzle
if "move_count" not in st.session_state:
    st.session_state.move_count = 0

defaults = {
    "state": GOAL,                         # current puzzle state (tuple)
    "history": [],                         # list of previous states (optional)
    "solution": None,                      # solver path if computed
    "sol_index": 0,                        # index into solution path
    "move_count": 0,                       # your move counter
    "start_time": None,                    # timestamp when timer started
    "auto_play": False,                    # whether solution autoplay is running
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Image slicing 
def crop_center_square(img: Image.Image) -> Image.Image:
    w,h = img.size
    m = min(w,h)
    left = (w-m)//2
    top = (h-m)//2
    return img.crop((left, top, left+m, top+m))

def slice_into_tiles(img: Image.Image):
    """Return dict mapping tile number (1..8) and 0->blank image"""
    img = crop_center_square(img)
    img = img.resize((450,450), Image.LANCZOS)  # 150x150 each
    tiles = {}
    size = img.size[0]//3
    num = 1
    for r in range(3):
        for c in range(3):
            box = (c*size, r*size, (c+1)*size, (r+1)*size)
            tile_img = img.crop(box)
            tiles[num] = tile_img
            num += 1
    # Tile style
    blank = Image.new("RGB", (size,size), (240,240,240))
    draw = ImageDraw.Draw(blank)
    draw.rectangle([1,1,size-2,size-2], outline=(200,200,200))
    tiles[0] = blank
    return tiles

# Streamlit app 
st.set_page_config(page_title="🧩 Puzzle Your Image! 🧩", layout="centered")
st.title("🧩 Puzzle Your Image! 🧩")

# Session state initialization
if "tiles" not in st.session_state:
    # load default image from an embedded small placeholder (a simple colored PIL created image)
    default = Image.new("RGB",(450,450),(180,200,230))
    draw = ImageDraw.Draw(default)
    draw.text((20,20),"Upload an image to make your puzzle", fill=(10,10,10))
    st.session_state.tiles = slice_into_tiles(default)
if "state" not in st.session_state:
    st.session_state.state = GOAL
if "history" not in st.session_state:
    st.session_state.history = []
if "solution" not in st.session_state:
    st.session_state.solution = None
if "sol_index" not in st.session_state:
    st.session_state.sol_index = 0
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

# Upload
uploaded = st.file_uploader("Upload image (jpg/png)", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.session_state.tiles = slice_into_tiles(img)
    st.session_state.state = GOAL
    st.session_state.solution = None
    st.session_state.sol_index = 0
    st.success("Image loaded and sliced. Start shuffling or play from the goal!")

# Controls
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("Shuffle pieces"):
        # perform many random legal moves from GOAL to guarantee solvable
        s = list(GOAL)
        moves = random.randint(20,60)
        last = None
        for _ in range(moves):
            nbs = neighbors(tuple(s))
            # avoid undo
            if last and len(nbs)>1:
                nbs = [nb for nb in nbs if nb != last]
            nxt = random.choice(nbs)
            last = tuple(s)
            s = list(nxt)
        st.session_state.state = tuple(s)
        st.session_state.history = []
        st.session_state.solution = None
        st.session_state.sol_index = 0
        st.session_state.move_count = 0
        st.session_state.start_time = time.time()

with col2:
    if st.button("Reset to goal"):
        st.session_state.state = GOAL
        st.session_state.history = []
        st.session_state.solution = None
        st.session_state.sol_index = 0
        st.session_state.move_count = 0
        st.session_state.start_time = time.time()

with col3:
    if st.button("Solve (A* - optimal)"):
        if not is_solvable(st.session_state.state):
            st.error("This configuration is not solvable!")
        else:
            with st.spinner("Running A*..."):
                t0 = time.time()
                path = astar(st.session_state.state, GOAL)
                t1 = time.time()
            if path is None:
                st.error("A* failed to find a solution within limits.")
            else:
                st.session_state.solution = path
                st.session_state.sol_index = 0
                st.success(f"Found solution in {len(path)-1} moves (time {t1-t0:.2f}s).")
with col4:
    import time
    if "move_count" not in st.session_state:
        st.session_state.move_count = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None

# Show status
st.markdown(f"**Current state (solvable: {'Yes' if is_solvable(st.session_state.state) else 'No'})**")
moves_so_far = st.session_state.move_count
if st.session_state.start_time:
    elapsed = int(time.time() - st.session_state.start_time)
else:
    elapsed = 0
minutes, seconds = divmod(elapsed, 60)

st.write(f"**Moves made:** {moves_so_far}")
st.write(f"**Time elapsed:** {minutes:02d}:{seconds:02d}")

# Puzzle grid display
state = list(st.session_state.state)
tiles = st.session_state.tiles

for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        idx = rc_to_index(r, c)
        tile_num = state[idx]

        with cols[c]:
            # Render the tile as a clickable button with its image
            if st.button(
                f"{tile_num}",  # key label (unique per tile)
                key=f"tile_{idx}",
            ):
                zero_idx = state.index(0)
                zr, zc = index_to_rc(zero_idx)
                tr, tc = index_to_rc(idx)

                # Only move if tile is adjacent to blank
                if abs(zr - tr) + abs(zc - tc) == 1:
                    state[zero_idx], state[idx] = state[idx], state[zero_idx]
                    st.session_state.history.append(tuple(state))
                    st.session_state.state = tuple(state)
                    st.session_state.solution = None
                    st.session_state.move_count += 1

            # Always display the tile image (not clickable if it's the blank)
            st.image(tiles[tile_num], use_container_width=True)

# If a solution exists, provide stepper to walk through
if st.session_state.solution:
    st.markdown("---")
    st.subheader("Solution playback")
    sol = st.session_state.solution
    st.write(f"Total moves: {len(sol)-1}")
    col_a, col_b, col_c, col_d = st.columns([1,1,1,1])
    with col_a:
        if st.button("Prev"):
            st.session_state.sol_index = max(0, st.session_state.sol_index-1)
    with col_b:
        if st.button("Next"):
            st.session_state.sol_index = min(len(sol)-1, st.session_state.sol_index+1)
    with col_c:
        if st.button("Go to start"):
            st.session_state.sol_index = 0
    with col_d:
        if st.button("Go to end"):
            st.session_state.sol_index = len(sol)-1

    # Auto-play controls
    ap_col1, ap_col2 = st.columns([1,1])
    with ap_col1:
        if st.button("Auto-play"):
            st.session_state.auto_play = True
    with ap_col2:
        if st.button("Stop"):
            st.session_state.auto_play = False

    # show current solution step
    idx = st.session_state.sol_index
    st.write(f"Step {idx} / {len(sol)-1}")
    # draw the state images inline
    cur = sol[idx]
    cols = st.columns(3)
    for c in range(3):
        idx = r * 3 + c
        tile_num = state[idx]

        with cols[c]:
            if tile_num == 0:
                st.image(tiles[0], use_container_width=True)  # blank tile
            else:
                if st.image_button(tiles[tile_num], key=f"tile_{idx}", use_container_width=True):
                    zero_idx = state.index(0)
                    zr, zc = divmod(zero_idx, 3)
                    tr, tc = divmod(idx, 3)
                    if abs(zr - tr) + abs(zc - tc) == 1:
                        new_state = list(state)
                        new_state[zero_idx], new_state[idx] = new_state[idx], new_state[zero_idx]
                        st.session_state.state = tuple(new_state)
                        st.session_state.move_count += 1

    # autoplay mechanism: advance index and rerun while auto_play true
    if st.session_state.auto_play:
        # small delay and advance
        next_idx = min(len(sol)-1, st.session_state.sol_index+1)
        if next_idx != st.session_state.sol_index:
            st.session_state.sol_index = next_idx
            time.sleep(0.4)
            st.experimental_rerun()
        else:
            st.session_state.auto_play = False
            st.success("Reached final state.")

# Footer / small help
st.markdown("---")
st.markdown("""
**Instructions**
- Upload an image to be cropped to center and sliced into 9 tiles.
- Shuffle to scramble the puzzle pieces (or manually move tiles by clicking Move buttons).
- Press *Solve (A\**)* to compute the optimal solution; then step through or auto-play.
""")

