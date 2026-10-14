/* Short end-of-session survey: "Did you like it?" + "Did you have any problems?"
   Usage: showSurvey(document.getElementById('survey'), sessionId, 'result' | 'error') */
(function () {
  const PROBLEMS = [
    ['slow', 'The analysis took too long'],
    ['error', 'I got an error / the analysis did not complete'],
    ['irrelevant', 'The recommended projects were irrelevant to my profile'],
    ['interface', 'I found the interface hard to understand'],
    ['cv_link', 'Problem finding/entering the Europass CV link'],
    ['other', 'Other'],
  ];

  const css = `
  .survey-card{background:#fff;border:2px solid #e6e3f7;border-radius:14px;padding:22px;margin-top:24px;text-align:left}
  .survey-card h3{margin:0 0 4px;color:#333;font-size:1.15em}
  .survey-card .survey-sub{color:#777;font-size:.9em;margin-bottom:16px}
  .survey-question{margin:14px 0 8px;font-weight:600;color:#333}
  .survey-options{display:flex;gap:10px;flex-wrap:wrap}
  .survey-btn{flex:1 1 140px;padding:11px 14px;border:2px solid #ddd;background:#fafafa;border-radius:10px;cursor:pointer;font-size:.98em;font-family:inherit;transition:all .15s}
  .survey-btn:hover{border-color:#667eea}
  .survey-btn.selected{border-color:#667eea;background:#eef0fd;color:#3f46b8;font-weight:600}
  .survey-problems{display:none;margin-top:10px;padding:12px;background:#fff8f0;border-radius:10px}
  .survey-problems label{display:flex;align-items:center;gap:8px;margin:6px 0;font-weight:400;cursor:pointer;color:#444}
  .survey-card textarea{width:100%;min-height:70px;margin-top:6px;padding:10px;border:2px solid #e0e0e0;border-radius:10px;font-family:inherit;font-size:.95em;resize:vertical}
  .survey-submit{margin-top:14px;width:100%;padding:13px;border:none;border-radius:10px;color:#fff;font-weight:600;font-size:1em;cursor:pointer;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}
  .survey-submit:disabled{opacity:.5;cursor:not-allowed}
  .survey-status{margin-top:10px;font-size:.92em}
  .survey-thanks{padding:18px;text-align:center;color:#155724;background:#d4edda;border-radius:10px;font-weight:600}
  `;
  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  window.showSurvey = function (container, sessionId, page) {
    if (!container || !sessionId) return;
    const status = { liked: null, had_problem: null };
    container.innerHTML = `
      <div class="survey-card">
        <h3>📝 Quick feedback (10 seconds)</h3>
        <div class="survey-sub">Your answers are recorded anonymously and used in our research only as aggregate numbers.</div>
        <div class="survey-question">1. Did you like the system?</div>
        <div class="survey-options" data-question="liked">
          <button type="button" class="survey-btn" data-value="true">👍 Yes, I liked it</button>
          <button type="button" class="survey-btn" data-value="false">👎 No</button>
        </div>
        <div class="survey-question">2. Did you have any problems while using it?</div>
        <div class="survey-options" data-question="had_problem">
          <button type="button" class="survey-btn" data-value="false">✅ No, it worked fine</button>
          <button type="button" class="survey-btn" data-value="true">⚠️ Yes, I had a problem</button>
        </div>
        <div class="survey-problems">
          <div style="font-size:.9em;color:#666;margin-bottom:4px">What kind of problem? (you can select more than one)</div>
          ${PROBLEMS.map(([k, t]) => `<label><input type="checkbox" value="${k}"> ${t}</label>`).join('')}
        </div>
        <div class="survey-question" style="font-weight:400;color:#555">Anything you want to add (optional)</div>
        <textarea maxlength="1000" placeholder="Suggestions, problem details, etc. Please do not write personal information."></textarea>
        <button type="button" class="survey-submit" disabled>Submit</button>
        <div class="survey-status"></div>
      </div>`;

    const submitBtn = container.querySelector('.survey-submit');
    const problemBox = container.querySelector('.survey-problems');
    const statusText = container.querySelector('.survey-status');

    // On the error page, preselect "I had a problem" and the "error" type
    if (page === 'error') {
      status.had_problem = true;
      container.querySelector('[data-question="had_problem"] [data-value="true"]').classList.add('selected');
      problemBox.style.display = 'block';
      problemBox.querySelector('input[value="error"]').checked = true;
      submitBtn.disabled = false;
    }

    container.querySelectorAll('.survey-options').forEach(group => {
      group.addEventListener('click', e => {
        const btn = e.target.closest('.survey-btn');
        if (!btn) return;
        group.querySelectorAll('.survey-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        status[group.dataset.question] = btn.dataset.value === 'true';
        problemBox.style.display = status.had_problem ? 'block' : 'none';
        submitBtn.disabled = status.liked === null && status.had_problem === null;
      });
    });

    submitBtn.addEventListener('click', async () => {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Sending...';
      const problem_types = Array.from(problemBox.querySelectorAll('input:checked')).map(i => i.value);
      try {
        const r = await fetch('/survey', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            liked: status.liked,
            had_problem: status.had_problem,
            problem_types: status.had_problem ? problem_types : [],
            comment: container.querySelector('textarea').value,
            page: page,
          }),
        });
        const d = await r.json();
        if (!r.ok || !d.success) throw new Error(d.error || 'Could not be sent');
        container.querySelector('.survey-card').innerHTML =
          '<div class="survey-thanks">🙏 Thank you! Your feedback has been saved.</div>';
      } catch (err) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Try again';
        statusText.style.color = '#c33';
        statusText.textContent = 'Could not be sent: ' + err.message;
      }
    });
  };
})();
