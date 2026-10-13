'use strict';
const form = document.querySelector('#loan-form');
const steps = [...document.querySelectorAll('.form-step')];
const nextButton = document.querySelector('#next-button');
const backButton = document.querySelector('#back-button');
const predictButton = document.querySelector('#predict-button');
const errorBanner = document.querySelector('#form-error');
const outlook = document.querySelector('#outlook-card');
const fieldNames = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'];
const numericNames = fieldNames.slice(0, 5);
let currentStep = 0;
let submitting = false;
let probabilityAnimation;

function clearErrors() {
  errorBanner.hidden = true;
  errorBanner.textContent = '';
  form.querySelectorAll('.invalid').forEach(el => el.classList.remove('invalid'));
  form.querySelectorAll('[aria-invalid]').forEach(el => el.removeAttribute('aria-invalid'));
  form.querySelectorAll('.field-error').forEach(el => { el.textContent = ''; });
}
function fieldError(name, message) {
  const input = form.elements.namedItem(name);
  if (!input) return;
  input.closest('.field').classList.add('invalid');
  input.setAttribute('aria-invalid', 'true');
  document.getElementById(`${name}-error`).textContent = message;
}
function showStep(index, focus = true) {
  currentStep = index;
  steps.forEach((step, i) => { step.hidden = i !== index; });
  document.querySelectorAll('.stepper li').forEach((el, i) => {
    el.classList.toggle('current', i === index);
    el.classList.toggle('complete', i < index);
    if (i === index) el.setAttribute('aria-current', 'step');
    else el.removeAttribute('aria-current');
  });
  document.querySelector('#step-count').textContent = `STEP 0${index + 1} / 03`;
  nextButton.hidden = index === 2;
  predictButton.hidden = index !== 2;
  backButton.hidden = index === 0;
  document.querySelector('#form-security').hidden = index !== 0;
  if (focus) steps[index].querySelector('input,select').focus();
}
function validateStep(index) {
  let firstInvalid = null;
  steps[index].querySelectorAll('input,select').forEach(input => {
    let message = '';
    if (!input.value.trim()) message = 'This field is required.';
    else if (!input.validity.valid) {
      message = input.type === 'number' ? 'Enter a valid number within the allowed range.' : 'Select an available option.';
      if (input.validity.rangeUnderflow) message = Number(input.min) > 0 ? 'Enter a number greater than zero.' : 'Enter zero or a positive number.';
      if (input.validity.stepMismatch) message = 'Enter a whole number of months.';
    }
    if (message) { fieldError(input.name, message); firstInvalid ||= input; }
  });
  if (firstInvalid) { showStep(index, false); firstInvalid.focus(); }
  return !firstInvalid;
}
nextButton.addEventListener('click', () => {
  if (submitting) return;
  clearErrors();
  if (validateStep(currentStep)) showStep(currentStep + 1);
});
backButton.addEventListener('click', () => { if (!submitting) { clearErrors(); showStep(currentStep - 1); } });
form.addEventListener('input', event => {
  const input = event.target;
  if (!input.name) return;
  input.closest('.field').classList.remove('invalid');
  input.removeAttribute('aria-invalid');
  document.getElementById(`${input.name}-error`).textContent = '';
});
function setState(state) {
  for (const name of ['empty', 'loading', 'result']) document.getElementById(`${name}-state`).hidden = name !== state;
  outlook.setAttribute('aria-busy', String(state === 'loading'));
}
function setBusy(busy) {
  submitting = busy;
  form.querySelectorAll('input,select,button').forEach(el => { el.disabled = busy; });
  document.querySelector('#reset-button').disabled = busy;
  predictButton.textContent = busy ? 'Evaluating…' : 'Get my estimate ↗';
}
function renderResult(prediction, values) {
  const approved = prediction.decision === 'approved';
  outlook.classList.toggle('rejected', !approved);
  document.querySelector('#result-icon').textContent = approved ? '✓' : '↘';
  document.querySelector('#result-title').textContent = prediction.title;
  document.querySelector('#result-note').textContent = prediction.disclaimer;
  const summary = document.querySelector('#result-summary');
  summary.replaceChildren();
  const rows = [['Loan amount', `${values.LoanAmount.toLocaleString()} dataset units`], ['Loan term', `${values.Loan_Amount_Term.toLocaleString()} months`], ['Model', prediction.model]];
  for (const [name, value] of rows) {
    const dt = document.createElement('dt'); dt.textContent = name;
    const dd = document.createElement('dd'); dd.textContent = value;
    summary.append(dt, dd);
  }
  const probability = prediction.approval_probability;
  document.querySelector('#probability-section').hidden = probability === null;
  cancelAnimationFrame(probabilityAnimation);
  setState('result');
  if (probability !== null) {
    const target = probability * 100;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const start = performance.now();
    const tick = now => {
      const progress = reduced ? 1 : Math.max(0, Math.min((now - start) / 700, 1));
      const value = target * (1 - Math.pow(1 - progress, 3));
      document.querySelector('#probability-value').textContent = `${value.toFixed(1)}%`;
      document.querySelector('#probability-bar').value = value;
      if (progress < 1) probabilityAnimation = requestAnimationFrame(tick);
    };
    probabilityAnimation = requestAnimationFrame(tick);
  }
  document.querySelector('#result-state').focus({ preventScroll: true });
  if (window.innerWidth < 981) outlook.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (submitting) return;
  clearErrors();
  if (currentStep < 2) { if (validateStep(currentStep)) showStep(currentStep + 1); return; }
  for (let i = 0; i < steps.length; i++) if (!validateStep(i)) return;
  const values = Object.fromEntries(fieldNames.map(name => [name, numericNames.includes(name) ? Number(form.elements.namedItem(name).value) : form.elements.namedItem(name).value]));
  setBusy(true);
  outlook.classList.remove('rejected');
  setState('loading');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values), signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response. Please try again.'); }
    if (!response.ok || !data.ok) {
      if (data.error?.fields) {
        const invalidFields = Object.entries(data.error.fields);
        invalidFields.forEach(([name, message]) => fieldError(name, message));
        const first = invalidFields.find(([name]) => fieldNames.includes(name));
        if (first) {
          const input = form.elements.namedItem(first[0]);
          showStep(steps.indexOf(input.closest('.form-step')), false);
        }
      }
      throw new Error(data.error?.message || 'The estimate could not be completed. Please try again.');
    }
    const p = data.prediction;
    if (!p || !['approved', 'rejected'].includes(p.decision) || typeof p.title !== 'string' ||
        !(p.approval_probability === null || (typeof p.approval_probability === 'number' && Number.isFinite(p.approval_probability) && p.approval_probability >= 0 && p.approval_probability <= 1))) {
      throw new Error('The server returned an invalid prediction. Please try again.');
    }
    renderResult(p, values);
  } catch (error) {
    setState('empty');
    errorBanner.textContent = error.name === 'AbortError' ? 'The request timed out. Please check the server and try again.' : error instanceof TypeError ? 'Unable to reach the server. Check your connection and try again.' : error.message;
    errorBanner.hidden = false;
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
});
document.querySelector('#reset-button').addEventListener('click', () => {
  if (submitting) return;
  cancelAnimationFrame(probabilityAnimation);
  form.reset(); clearErrors(); showStep(0); setState('empty');
  outlook.classList.remove('rejected');
  document.querySelector('#application').scrollIntoView({ behavior: 'smooth', block: 'start' });
});
for (const trigger of document.querySelectorAll('[data-dialog]')) trigger.addEventListener('click', () => document.getElementById(trigger.dataset.dialog).showModal());
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
}
async function checkHealth() {
  const status = document.querySelector('#model-status');
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (!response.ok || !(await response.json()).ok) throw new Error('Unavailable');
    status.replaceChildren(document.createElement('i'), document.createTextNode('Model connected'));
  } catch {
    status.classList.add('unavailable');
    status.replaceChildren(document.createElement('i'), document.createTextNode('Model unavailable'));
  }
}
checkHealth();
