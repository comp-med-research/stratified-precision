const canvas = document.getElementById("network-bg");
const ctx    = canvas.getContext("2d");

const N        = 70;
const MAX_DIST = 200;
const HUB_N    = Math.floor(N * 0.12);

// All colours defined per-theme — no globals
const THEMES = {
  dark: {
    top: "#0a0f1e", bot: "#0d1530",
    node: { r: 91,  g: 141, b: 239 },   // blue
    hub:  { r: 76,  g: 175, b: 125 },   // green
    edgeAlpha: 0.28, nodeAlpha: [0.90, 0.72],
  },
  light: {
    top: "#ffffff", bot: "#ffffff",
    node: { r: 91,  g: 141, b: 239 },   // same blue as dark
    hub:  { r: 76,  g: 175, b: 125 },   // same green as dark
    edgeAlpha: 0.22, nodeAlpha: [0.85, 0.65],
  },
};

let W, H, particles;

function theme() {
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? THEMES.dark : THEMES.light;
}

function rgba(col, a) {
  return `rgba(${col.r},${col.g},${col.b},${a.toFixed(3)})`;
}

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;

  particles = Array.from({ length: N }, (_, i) => {
    const hub = i < HUB_N;
    const spd = (hub ? 0.15 : 0.3) * (0.6 + Math.random() * 0.8);
    const ang = Math.random() * Math.PI * 2;
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: Math.cos(ang) * spd,
      vy: Math.sin(ang) * spd,
      r:  hub ? 3.5 + Math.random() * 3 : 1.5 + Math.random() * 2,
      hub,
      phase: Math.random() * Math.PI * 2,
    };
  });
}

function animate(ts) {
  const t  = ts * 0.001;
  const th = theme();   // reads data-theme every frame → instant toggle

  // Background gradient
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, th.top);
  bg.addColorStop(1, th.bot);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Move & wrap
  for (const p of particles) {
    p.x += p.vx;  p.y += p.vy;
    if (p.x < -60)    p.x = W + 60;
    if (p.x > W + 60) p.x = -60;
    if (p.y < -60)    p.y = H + 60;
    if (p.y > H + 60) p.y = -60;
  }

  // Edges
  for (let i = 0; i < particles.length; i++) {
    const a = particles[i];
    for (let j = i + 1; j < particles.length; j++) {
      const b   = particles[j];
      const d   = Math.hypot(a.x - b.x, a.y - b.y);
      const thr = MAX_DIST * ((a.hub || b.hub) ? 1.5 : 1);
      if (d < thr) {
        const col   = a.hub ? th.hub : (b.hub ? th.hub : th.node);
        const alpha = Math.pow(1 - d / thr, 1.8) * th.edgeAlpha;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = rgba(col, alpha);
        ctx.lineWidth   = alpha * 6;
        ctx.stroke();
      }
    }
  }

  // Nodes + glow
  for (const p of particles) {
    const col    = p.hub ? th.hub : th.node;
    const pulse  = p.hub ? 1 + Math.sin(t * 1.1 + p.phase) * 0.2 : 1;
    const r      = p.r * pulse;
    const alphas = th.nodeAlpha;

    // Glow
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 5.5);
    g.addColorStop(0, rgba(col, 0.15));
    g.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 5.5, 0, Math.PI * 2);
    ctx.fill();

    // Core
    ctx.fillStyle = rgba(col, p.hub ? alphas[0] : alphas[1]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  requestAnimationFrame(animate);
}

window.addEventListener("resize", resize);
resize();
requestAnimationFrame(animate);
