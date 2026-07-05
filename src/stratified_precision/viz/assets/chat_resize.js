(function () {
    var MIN_W = 280, MAX_W = 900;

    // Event delegation — survives Dash/React re-renders because listener is on document
    document.addEventListener('mousedown', function (e) {
        var handle = e.target.closest && e.target.closest('#chat-resize-handle');
        if (!handle) return;

        var panel = document.getElementById('chat-panel');
        if (!panel) return;

        var startX = e.clientX;
        var startW = panel.offsetWidth;

        e.preventDefault();
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';

        function onMove(ev) {
            var w = Math.min(MAX_W, Math.max(MIN_W, startW + (startX - ev.clientX)));
            panel.style.width = w + 'px';
        }

        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}());

// Auto-scroll chat to bottom whenever messages change
new MutationObserver(function () {
    var msgs = document.getElementById('chat-messages');
    if (msgs) msgs.scrollTop = msgs.scrollHeight;
}).observe(document.documentElement, { childList: true, subtree: true });
