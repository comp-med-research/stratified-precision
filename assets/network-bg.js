(function () {
  var NODE_COLOR = "91, 141, 239";
  var HUB_COLOR  = "76, 175, 125";
  var N_NODES    = 65;
  var MAX_DIST   = 200;
  var SPEED      = 0.3;

  // ── Create canvas immediately, insert into body as soon as it exists ──
  var canvas = document.createElement("canvas");
  canvas.style.cssText = [
    "position:fixed", "top:0", "left:0",
    "width:100%", "height:100%",
    "z-index:0", "pointer-events:none", "display:block",
  ].join(";");

  var ctx, nodes;

  function boot() {
    document.body.insertBefore(canvas, document.body.firstChild);
    ctx = canvas.getContext("2d");
    resize();
    spawnNodes();
    window.addEventListener("resize", function () { resize(); spawnNodes(); });
    requestAnimationFrame(draw);
  }

  if (document.body) {
    boot();
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }

  // ── Sizing ────────────────────────────────────────────────────────────
  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  // ── Nodes ─────────────────────────────────────────────────────────────
  function spawnNodes() {
    nodes = [];
    var nHubs = Math.floor(N_NODES * 0.12);
    for (var i = 0; i < N_NODES; i++) {
      var hub = i < nHubs;
      nodes.push({
        x:     Math.random() * canvas.width,
        y:     Math.random() * canvas.height,
        vx:    (Math.random() - 0.5) * SPEED * (hub ? 0.45 : 1),
        vy:    (Math.random() - 0.5) * SPEED * (hub ? 0.45 : 1),
        r:     hub ? Math.random() * 3 + 4.5 : Math.random() * 1.8 + 1.5,
        hub:   hub,
        phase: Math.random() * Math.PI * 2,
        col:   hub ? HUB_COLOR : NODE_COLOR,
      });
    }
  }

  // ── Draw loop ─────────────────────────────────────────────────────────
  function draw(ts) {
    requestAnimationFrame(draw);
    var t = (ts || 0) * 0.001;
    var W = canvas.width, H = canvas.height;

    // Background gradient
    var bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, "#0a0f1e");
    bg.addColorStop(1, "#0d1530");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Move
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.x += n.vx;  n.y += n.vy;
      if (n.x < -60)   n.x = W + 60;
      if (n.x > W + 60) n.x = -60;
      if (n.y < -60)   n.y = H + 60;
      if (n.y > H + 60) n.y = -60;
    }

    // Edges
    for (var i = 0; i < nodes.length; i++) {
      var a = nodes[i];
      for (var j = i + 1; j < nodes.length; j++) {
        var b = nodes[j];
        var dx = a.x - b.x, dy = a.y - b.y;
        var d  = Math.sqrt(dx * dx + dy * dy);
        var th = MAX_DIST * ((a.hub || b.hub) ? 1.5 : 1);
        if (d < th) {
          var alpha = Math.pow(1 - d / th, 1.8) * 0.3;
          ctx.strokeStyle = "rgba(" + NODE_COLOR + "," + alpha + ")";
          ctx.lineWidth = alpha * 1.5;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Nodes + glow
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var pulse = n.hub ? 1 + Math.sin(t * 1.1 + n.phase) * 0.2 : 1;
      var r = n.r * pulse;

      var g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 5.5);
      g.addColorStop(0, "rgba(" + n.col + ",0.2)");
      g.addColorStop(1, "rgba(" + n.col + ",0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r * 5.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "rgba(" + n.col + "," + (n.hub ? 0.9 : 0.72) + ")";
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
})();
