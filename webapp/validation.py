"""Strict JSON contract for the nine verified model features."""
import math
from loan_pipeline.config import FEATURES, DOMAINS

NUMBERS = {
    'ApplicantIncome': (0, False), 'CoapplicantIncome': (0, False),
    'LoanAmount': (0, True), 'Loan_Amount_Term': (0, True), 'Credit_History': (0, False),
}


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, {'form': 'Send a JSON object containing applicant details.'}
    errors, clean = {}, {}
    if set(payload) - set(FEATURES):
        errors['form'] = 'Unexpected fields were supplied. Use only the application fields.'
    for name in FEATURES:
        value = payload.get(name)
        if value is None or value == '':
            errors[name] = 'This field is required.'
        elif name in NUMBERS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors[name] = 'Enter a valid number.'
                continue
            try:
                valid = math.isfinite(value) and abs(value) <= 1e12
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                errors[name] = 'Enter a finite number no greater than 1 trillion.'
            elif value < 0 or (NUMBERS[name][1] and value == 0):
                errors[name] = 'Enter a number greater than zero.' if NUMBERS[name][1] else 'Enter zero or a positive number.'
            elif name == 'Credit_History' and value not in (0, 1):
                errors[name] = 'Select a credit history value of 0 or 1.'
            elif name == 'Loan_Amount_Term' and value != int(value):
                errors[name] = 'Enter a whole number of months.'
            else:
                clean[name] = float(value)
        elif not isinstance(value, str) or value not in DOMAINS[name]:
            errors[name] = 'Select one of the available options.'
        else:
            clean[name] = value
    return (clean if not errors else None), errors
