/* Copyright (c) Meta Platforms, Inc. and affiliates. */
// <truchet-pattern> — concentric quarter-arc Truchet tiles (after bookofshapes.com "Quarter Arc Truchet").
// Each tile holds a fan of `arcs` concentric quarter-arcs around one corner, i.e. a 90° turn joining
// the two edges at that corner. A "snake" is a chain of tiles whose fans meet on shared edges: a
// ribbon of parallel strands winding through the grid, drawn in one colour. The canvas starts with
// snakes of SNAKE_LEN tiles laid as random walks, leaving some tiles empty.
//
// With `crawl`, snakes move like the game Snake, driven by scroll: the head sweeps into the next
// tile (turning left or right there) while the tail sweeps out of the last one. Each snake starts at
// its own point in the scroll and then never stops until it has left the canvas. To guarantee that,
// a snake plans its whole route out when it starts, reserving every tile for the ticks it will
// occupy it; later snakes plan around those reservations, and a snake that cannot find a clear
// route yet waits before starting. Whenever the canvas is emptier than it started, new snakes are
// added the same way: some enter from outside the edges, some hatch in an empty tile and grow in
// behind their head. Speeds vary per snake and a few are fast. Everything is a pure function of
// the scroll position, so scrolling up rewinds.
//
// Attributes: cols, rows, arcs, ratio (band fill 0..1), seed (omit for a new layout per load),
// colors (space-separated CSS colours), crawl (px of scrolling per tile for a normal snake; omit
// for a static pattern), start-now (share of snakes ready on the first scrolled pixel; default
// START_NOW_P).

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const S = 100;            // tile size in SVG units
const EMPTY = 0.35;       // fraction of tiles left empty at generation
const SLOW_TICKS = 3;     // ticks per tile for a normal snake (a tick is crawl / SLOW_TICKS px)
const FAST_TICKS = 1;     // ticks per tile for a fast snake
const FAST_P = 0.2;       // fraction of fast snakes
const SPEED_JITTER = 0.25; // each snake's ticks per tile vary by up to ±25%
const START_NOW_P = 0.5;  // share of canvas snakes ready to start as soon as scrolling begins
const START_SPREAD = 24;  // the rest become ready at a random tick in [1, START_SPREAD)
const SPAWN_TRIES = 3;    // max attempts per tick to add snakes while the canvas is below its starting density
const HATCH_P = 0.5;      // share of new snakes that hatch inside the canvas (the rest enter from an edge)
const SNAKE_LEN = [2, 4]; // length range of snakes, at initialisation and when added later
const MAX_NODES = 4000;   // route search budget per planning attempt

const f = v => +v.toFixed(2);
const crawlers = new Set();
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  addEventListener("scroll", () => requestAnimationFrame(() => crawlers.forEach(el => el.update())),
                   { passive: true });
}

// A tile's corner c = hx + 2*vy (hx: 0 left / 1 right, vy: 0 top / 1 bottom). Its fan touches the
// horizontal and vertical edges on the corner's side; fanDirs gives the neighbour directions across
// those two edges, [horizontal-edge dir, vertical-edge dir].
const fanDirs = c => [[0, c >> 1 ? 1 : -1], [c & 1 ? 1 : -1, 0]];

// The two corners for a tile entered moving in direction [dx, dy] (the fan must touch the entry
// edge), each with the direction the snake leaves in.
const turns = ([dx, dy]) => [0, 1].map(s => {
  const c = dx ? (dx < 0 ? 1 : 0) + 2 * s : s + 2 * (dy < 0 ? 1 : 0);
  return [c, fanDirs(c)[dx ? 0 : 1]];
});

class TruchetPattern extends HTMLElement {
  connectedCallback() { this.setup(); this.update(); }
  disconnectedCallback() { crawlers.delete(this); }

  setup() {
    const num = (k, d) => +(this.getAttribute(k) ?? d);
    const cols = num("cols", 6), rows = num("rows", 6), n = num("arcs", 6), ratio = num("ratio", 0.1085);
    const rand = this.rand = mulberry32(num("seed", Math.floor(Math.random() * 2 ** 32)));
    this.tickPx = num("crawl", 0) / SLOW_TICKS;
    const startNow = num("start-now", START_NOW_P);
    this.colors = (this.getAttribute("colors") || "currentColor").trim().split(/\s+/);
    Object.assign(this, { cols, rows, n });
    this.w = ratio * S / n;                // band width
    this.d = (S - this.w) / (n - 1);       // band spacing; band k's centreline is at k*d + w/2

    // Lay snakes as random walks through empty tiles until the canvas is (1 - EMPTY) full. Each walk
    // starts with a random fan and turns randomly left or right; it stops early at the canvas edge
    // or an occupied tile, and walks shorter than SNAKE_LEN[0] are discarded.
    this.snakes = [];
    this.res = new Map();                  // "x,y" -> [[from, to, snake], ...] tick intervals
    const occupied = new Set(), want = Math.round((1 - EMPTY) * cols * rows);
    let filled = 0;
    for (let tries = 0; filled < want && tries < 50 * cols * rows; tries++) {
      const L = SNAKE_LEN[0] + Math.floor(rand() * (SNAKE_LEN[1] - SNAKE_LEN[0] + 1));
      const body = [{ x: Math.floor(rand() * cols), y: Math.floor(rand() * rows), c: Math.floor(rand() * 4) }];
      if (occupied.has(`${body[0].x},${body[0].y}`)) continue;
      let dir = fanDirs(body[0].c)[Math.floor(rand() * 2)];
      while (body.length < L) {
        const x = body[body.length - 1].x + dir[0], y = body[body.length - 1].y + dir[1];
        if (!this.inside(x, y) || occupied.has(`${x},${y}`) || body.some(b => b.x === x && b.y === y)) break;
        const [c, out] = turns(dir)[Math.floor(rand() * 2)];
        body.push({ x, y, c });
        dir = out;
      }
      if (body.length < SNAKE_LEN[0]) continue;
      const sn = this.addSnake(body, dir);   // body is tail -> head; dir is where the head points
      sn.ready = rand() < startNow ? 0 : 1 + Math.floor(rand() * (START_SPREAD - 1));
      for (const b of body) { occupied.add(`${b.x},${b.y}`); this.reserve(b.x, b.y, -Infinity, Infinity, sn); }
      filled += body.length;
    }
    this.target = filled;                  // canvas tiles to keep occupied
    this.simTick = 0;
    this.last = null;
    this.innerHTML = `<svg viewBox="0 0 ${cols * S} ${rows * S}" xmlns="http://www.w3.org/2000/svg" ` +
      `fill="none" stroke-width="${f(this.w)}"></svg>`;
    this.svg = this.firstChild;
    if (this.tickPx) crawlers.add(this);
  }

  inside(x, y) { return x >= 0 && y >= 0 && x < this.cols && y < this.rows; }

  addSnake(body, dir) {
    const sn = { body, dir, route: null, t0: Infinity, ready: 0,
                 color: Math.floor(this.rand() * this.colors.length),
                 P: (this.rand() < FAST_P ? FAST_TICKS : SLOW_TICKS) * (1 + SPEED_JITTER * (2 * this.rand() - 1)) };
    this.snakes.push(sn);
    return sn;
  }

  // Tick-interval reservations of canvas tiles; off-canvas tiles are never contended.
  reserve(x, y, a, b, sn) {
    if (!this.inside(x, y)) return;
    const k = `${x},${y}`;
    if (!this.res.has(k)) this.res.set(k, []);
    this.res.get(k).push([a, b, sn]);
  }
  release(sn) { for (const list of this.res.values()) for (let i = list.length; i--; ) if (list[i][2] === sn) list.splice(i, 1); }
  clear(x, y, a, b) {
    return !this.inside(x, y) || !(this.res.get(`${x},${y}`) || []).some(([c, d]) => a < d && c < b);
  }

  // Plan sn's route from tick t until its whole body has left the canvas. With P ticks per tile and
  // L body tiles, the k-th new tile is entered at t + k*P and freed once the tail has passed it, at
  // t + (k + L + 1)*P; body tile i (0 = tail) is freed at t + (i + 1)*P. Returns true on success.
  plan(sn, t) {
    const { body, P } = sn, L = body.length, route = [], visited = new Set();
    this.release(sn);
    body.forEach((b, i) => b.ghost || this.reserve(b.x, b.y, -Infinity, t + (i + 1) * P, sn));
    let nodes = 0;
    const dfs = (x, y, dir, k) => {
      const nx = x + dir[0], ny = y + dir[1];
      if (!this.inside(nx, ny)) {
        // The head is out: keep going away from the canvas until the tail is out too.
        for (let j = 0, px = x, py = y, dr = dir; j < L; j++) {
          const qx = px + dr[0], qy = py + dr[1], opts = turns(dr);
          const [c, out] = opts.find(([, o]) => !this.inside(qx + o[0], qy + o[1])) || opts[0];
          route.push({ x: qx, y: qy, c });
          px = qx; py = qy; dr = out;
        }
        return true;
      }
      const a = t + k * P, b = t + (k + L + 1) * P, key = `${nx},${ny}`;
      if (++nodes > MAX_NODES || visited.has(key) || !this.clear(nx, ny, a, b)) return false;
      visited.add(key);                    // routes never cross themselves, so they always lead out
      const opts = turns(dir);
      if (this.rand() < 0.5) opts.reverse();
      for (const [c, out] of opts) {
        this.reserve(nx, ny, a, b, sn);
        route.push({ x: nx, y: ny, c });
        if (dfs(nx, ny, out, k + 1)) return true;
        route.pop();
        this.res.get(key).pop();
      }
      visited.delete(key);
      return false;
    };
    const h = body[L - 1];
    if (dfs(h.x, h.y, sn.dir, 0)) { sn.route = route; sn.t0 = t; return true; }
    this.release(sn);
    if (Number.isFinite(sn.ready)) body.forEach(b => this.reserve(b.x, b.y, -Infinity, Infinity, sn));
    return false;
  }

  // One simulation tick: waiting snakes that are ready try to start, then new snakes are added
  // while the canvas has fewer occupied tiles than it started with.
  tick(t) {
    const rand = this.rand;
    let occupied = 0;
    for (const [k, list] of this.res) {    // drop expired reservations, count tiles occupied now
      const live = list.filter(([, b]) => b > t);
      if (live.length) this.res.set(k, live); else this.res.delete(k);
      occupied += live.some(([a]) => a <= t);
    }
    const waiting = this.snakes.filter(sn => sn.t0 === Infinity && sn.ready <= t).sort(() => rand() - 0.5);
    for (const sn of waiting) this.plan(sn, t);

    for (let tries = 0, deficit = this.target - occupied; deficit > 0 && tries < SPAWN_TRIES; tries++) {
      // A new snake's body starts as invisible "ghost" tiles behind its head: just outside an edge
      // when entering, or next to an empty canvas tile when hatching there.
      let x, y, dir;
      if (rand() < HATCH_P) {
        dir = [[1, 0], [-1, 0], [0, 1], [0, -1]][Math.floor(rand() * 4)];
        x = Math.floor(rand() * this.cols) - dir[0]; y = Math.floor(rand() * this.rows) - dir[1];
      } else {
        const side = Math.floor(rand() * 4), i = Math.floor(rand() * (side < 2 ? this.rows : this.cols));
        [x, y, dir] = [[-1, i, [1, 0]], [this.cols, i, [-1, 0]], [i, -1, [0, 1]], [i, this.rows, [0, -1]]][side];
      }
      const L = SNAKE_LEN[0] + Math.floor(rand() * (SNAKE_LEN[1] - SNAKE_LEN[0] + 1));
      const sn = this.addSnake(Array.from({ length: L }, () => ({ x, y, c: 0, ghost: true })), dir);
      sn.ready = NaN;                      // not a waiting canvas snake: start now or never
      if (this.plan(sn, t)) deficit -= L; else this.snakes.pop();
    }
  }

  // Fan at (x, y) with corner c, swept from the edge facing (tx, ty) over fraction `frac` of 90°.
  fan({ x, y, c, ghost }, toward, frac) {
    if (frac <= 0 || ghost || !this.inside(x, y)) return "";
    const hx = c & 1, vy = c >> 1, cx = (x + hx) * S, cy = (y + vy) * S, sx = hx ? -1 : 1, sy = vy ? -1 : 1;
    const fromH = !toward || toward.x === x;                  // toward a vertical neighbour
    const a0 = fromH ? 0 : Math.PI / 2, a1 = fromH ? frac * Math.PI / 2 : Math.PI / 2 * (1 - frac);
    const sweep = (sx * sy > 0) === (a1 > a0) ? 1 : 0;
    let p = "";
    for (let k = 0; k < this.n; k++) {
      const r = k * this.d + this.w / 2;
      p += `M${f(cx + sx * r * Math.cos(a0))} ${f(cy + sy * r * Math.sin(a0))}` +
           `A${f(r)} ${f(r)} 0 0 ${sweep} ${f(cx + sx * r * Math.cos(a1))} ${f(cy + sy * r * Math.sin(a1))}`;
    }
    return p;
  }

  update() {
    const s = this.tickPx ? Math.max(0, scrollY) / this.tickPx : 0;
    if (this.last === s) return;
    this.last = s;
    while (this.simTick <= Math.floor(s)) this.tick(this.simTick++);
    const paths = this.colors.map(() => "");
    for (const { body, route, t0, P, color } of this.snakes) {
      if (s < t0) {                        // not moving yet (or not yet entered: its body is off-canvas)
        for (const b of body) paths[color] += this.fan(b, null, 1);
        continue;
      }
      const e = (s - t0) / P, m = Math.floor(e), p = e - m, L = body.length;
      if (m >= route.length) continue;     // gone
      const seq = body.concat(route);      // before move m the body is seq[m .. m+L-1]; the head enters seq[m+L]
      for (let i = m + 1; i < m + L; i++) paths[color] += this.fan(seq[i], null, 1);
      paths[color] += this.fan(seq[m], seq[m + 1], 1 - p);            // tail retracts toward the body
      paths[color] += this.fan(seq[m + L], seq[m + L - 1], p);        // head sweeps in from the old end
    }
    this.svg.innerHTML = paths.map((d, ci) => d && `<path d="${d}" stroke="${this.colors[ci]}"/>`).join("");
  }
}

customElements.define("truchet-pattern", TruchetPattern);
