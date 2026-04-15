// BibTeX copy button
(function() {
  var btn = document.getElementById('bibtex-copy-btn');
  if (!btn) return;
  btn.addEventListener('click', function() {
    var content = document.getElementById('bibtex-content').innerText;
    var showSuccess = function() {
      var icon = btn.querySelector('.icon i');
      var text = btn.querySelector('span:last-child');
      icon.className = 'fas fa-check';
      text.innerText = 'Copied';
      btn.classList.remove('is-light');
      btn.classList.add('is-success');
      setTimeout(function() {
        icon.className = 'fas fa-copy';
        text.innerText = 'Copy';
        btn.classList.remove('is-success');
        btn.classList.add('is-light');
      }, 2000);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(content).then(showSuccess).catch(function() { fallbackCopy(content); });
    } else {
      fallbackCopy(content);
    }
    function fallbackCopy(text) {
      var textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      textArea.style.top = "0";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try { if (document.execCommand('copy')) showSuccess(); } catch (err) {}
      document.body.removeChild(textArea);
    }
  });
})();
